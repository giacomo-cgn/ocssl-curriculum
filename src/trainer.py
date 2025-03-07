import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler

from .utils import UnsupervisedDataset
from .transforms import get_transforms
from .ssl_models import AbstractSSLModel
from .strategies import AbstractStrategy
from .optims import init_optim
from .probing import exec_probing


class Trainer():

    def __init__(self,
                 ssl_model: AbstractSSLModel = None,
                 strategy: AbstractStrategy = None,
                 optim: str = 'SGD',
                 lr: float = 0.01,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 lars_eta: float = 0.005,
                 train_mb_size: int = 32,
                 mb_passes: int = 1,
                 device = 'cpu',
                 dataset_name: str = 'cifar100',
                 save_pth: str  = None,
                 save_model: bool = False,
                 online_transforms: bool = True,
                 num_views: int = 2
               ):
        
        if ssl_model is None:
            raise Exception(f'A SSL model is requred')            

        self.ssl_model = ssl_model
        self.strategy = strategy
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lars_eta = lars_eta
        self.train_mb_size = train_mb_size
        self.mb_passes = mb_passes
        self.device = device
        self.dataset_name = dataset_name
        self.save_pth = save_pth
        self.save_model = save_model
        self.online_transforms = online_transforms
        self.num_views = num_views # == 2 for most Instance Discrimination methods, but can vary e.g. EMP

        self.model_and_strategy_name = self.strategy.get_name() + '_' + self.ssl_model.get_name()

        # Set up transforms
        self.transforms = get_transforms(dataset=self.dataset_name, n_crops=self.num_views, online_transforms=self.online_transforms)

        # List of params to optimize
        params_to_optimize = self.ssl_model.get_params() + self.strategy.get_params()

        # Set up optimizer
        self.optimizer = init_optim(optim, params_to_optimize, lr=self.lr, momentum=self.momentum,
                                    weight_decay=self.weight_decay, lars_eta=self.lars_eta)


        if self.save_pth is not None:
            # Save model configuration
            with open(self.save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- TRAINER CONFIG ----\n')
                f.write(f'optim: {optim}\n') 
                f.write(f'Learning Rate: {self.lr}\n')
                f.write(f'optim-momentum: {self.momentum}\n')
                f.write(f'weight_decay: {self.weight_decay}\n')
                if optim == 'lars':
                    f.write(f'lars_eta: {self.lars_eta}\n')
                f.write(f'num_views: {self.num_views}\n')
                f.write(f'train_mb_size: {self.train_mb_size}\n')


                # Write loss file column names
                with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                    f.write('loss,tr_step,mb_pass\n')


    def train_experience(self, 
                         dataset,
                         exp_tr_steps: int,
                         exp_idx: int,
                         before_tr_steps: int = 0,
                         eval_every_steps: int = 0,
                         eval_idx: int = 0,
                         intermediate_eval_dict: dict = {"status": False}, # Set to True to evaluate model at intermediate steps, contains vars for intermediate eval
                         ):
        # Prepare data
        if self.online_transforms:
            exp_data = UnsupervisedDataset(dataset)
        else:
            exp_data = UnsupervisedDataset(dataset, transforms=self.transforms)
        sampler = RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        data_loader = DataLoader(exp_data, batch_size=self.train_mb_size,  sampler=sampler, num_workers=8)

        self.ssl_model.train()
        self.strategy.train()

        self.strategy.before_experience()
        
        for tr_step_idx in tqdm(range(exp_tr_steps)):
            upto_tr_step_idx = before_tr_steps + tr_step_idx
            
            stream_mbatch = data_loader.__iter__().__next__()
            if self.online_transforms:
                stream_mbatch = stream_mbatch.to(self.device)
            else:
                stream_mbatch = [x.to(self.device) for x in stream_mbatch]

            stream_mbatch = self.strategy.before_mb_passes(stream_mbatch)

            for k in range(self.mb_passes):
                # Apply strategy modifications before forward pass (e.g. concat replay samples from buffer)
                mbatch = self.strategy.before_forward(stream_mbatch)

                # Apply transforms, obtains a list of tensors, each containing 1 view for every sample in the mbatch
                if self.online_transforms:
                    x_views_list = self.transforms(mbatch)
                else:
                    x_views_list = mbatch

                x_views_list = self.strategy.after_transforms(x_views_list)

                # Forward pass of SSL model (z: projector features, e: encoder features)
                loss, z_list, e_list = self.ssl_model(x_views_list)

                # Strategy after forward pass
                loss_strategy = self.strategy.after_forward(x_views_list, loss, z_list, e_list)

                if loss_strategy is not None:
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss_strategy.backward()
                    self.optimizer.step()

                self.ssl_model.after_backward()
                self.strategy.after_backward()

                # Save loss, exp_idx, epoch, mb_idx and k in csv
                if self.save_pth is not None and loss_strategy is not None:
                    with open(os.path.join(self.save_pth, 'pretr_loss.csv'), 'a') as f:
                        f.write(f'{loss_strategy.item()},{upto_tr_step_idx},{k}\n')

                # Check if have to evaluate IID model
                if intermediate_eval_dict["status"]:
                    if (upto_tr_step_idx+1) % eval_every_steps == 0:
                        current_classes = exp_data.get_classes()
                        exec_probing(kwargs=intermediate_eval_dict["kwargs"], probes=intermediate_eval_dict["probes"],
                                        probing_benchmark=intermediate_eval_dict["benchmark"], encoder=self.ssl_model.get_encoder_for_eval(), 
                                        pretr_exp_idx=eval_idx, save_pth=self.save_pth, device=self.device,
                                        prev_classes=intermediate_eval_dict["prev_classes"], curr_classes=current_classes)
                        eval_idx += 1
                    self.ssl_model.train()

            self.strategy.after_mb_passes()

        # Save model and optimizer state
        if self.save_model and self.save_pth is not None:
            chkpt_pth = os.path.join(self.save_pth, 'checkpoints')
            if not os.path.exists(chkpt_pth):
                os.makedirs(chkpt_pth)
            torch.save({
                'model_state_dict': self.ssl_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, os.path.join(chkpt_pth, f'model_exp{exp_idx}.pth'))

        return self.ssl_model, eval_idx, current_classes
