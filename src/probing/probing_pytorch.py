import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .abstract_probe import AbstractProbe
class ProbingPytorch(AbstractProbe):
    def __init__(self,                 
                 device: str = 'cpu',
                 mb_size: int = 512,
                 seed: int = 42,
                 config_save_pth: str = None,
                 dim_encoder_features: int = 512,
                 lr: float = 5e-2,
                 lr_patience: int = 5,
                 lr_factor: int = 3,
                 lr_min: float = 1e-4,
                 probing_epochs: int = 100
    ):
        
        self.device = device
        self.mb_size = mb_size
        self.seed = seed
        self.probe_type = "torch"
        self.dim_encoder_features = dim_encoder_features

        self.lr = lr
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.probing_epochs = probing_epochs

        self.criterion = nn.CrossEntropyLoss()

        if config_save_pth is not None:
            # Save model configuration
            with open(config_save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- PROBE CONFIG ----\n')
                f.write(f'Probing type: {self.probe_type}\n')
                f.write(f'Eval MB size: {mb_size}\n')
                f.write(f'Probing LR: {self.lr}\n')
                f.write(f'Probing lr patience: {self.lr_patience}\n')
                f.write(f'Probing lr factor: {self.lr_factor}\n')
                f.write(f'Probing lr min: {self.lr_min}\n')
                f.write(f'Probing epochs: {self.probing_epochs}\n')

    def get_name(self) -> str:
        return self.probe_type
    
    def probe(self,
              tr_activations: torch.Tensor,
              tr_labels: torch.Tensor,
              val_activations: torch.Tensor,
              val_labels: torch.Tensor,
              test_activations: torch.Tensor,
              test_labels: torch.Tensor,
              exp_idx: int = None, # Task index on which probing is executed, if None, we are in joint or upto probing
              save_file: str = None,
              ):
        
        if val_activations is None:
            raise ValueError("Validation dataset is required for PyTorch linear probing")
        
       
        self.exp_idx = exp_idx
        self.save_file = save_file

        if self.save_file is not None:
            with open(self.save_file, 'a') as f:
                # Write header for probing log file
                if not os.path.exists(self.save_file) or os.path.getsize(self.save_file) == 0:
                    if self.exp_idx is not None:
                        f.write('probing_exp_idx,val_acc,test_acc\n')
                    else:
                        f.write(f'val_acc,test_acc\n')
        
        with torch.no_grad():
            tr_activations = nn.functional.normalize(tr_activations.to(self.device))
            tr_labels = tr_labels.to(self.device)
            val_activations = nn.functional.normalize(val_activations.to(self.device))
            val_labels = val_labels.to(self.device)
            test_activations = nn.functional.normalize(test_activations.to(self.device))
            test_labels = test_labels.to(self.device)

            num_classes = len(torch.unique(tr_labels))
            if max(torch.unique(tr_labels)) > num_classes - 1:
                # If only a subset of labels, rename them in [0, num_class] range
                labels_set = set(torch.unique(tr_labels).tolist())
                new_labels = list(range(len(labels_set)))
                label_map = {k: v for k, v in zip(labels_set, new_labels)}
                tr_labels = torch.tensor([label_map[l.item()] for l in tr_labels], dtype=torch.long).to(self.device)
                val_labels = torch.tensor([label_map[l.item()] for l in val_labels], dtype=torch.long).to(self.device)
                test_labels = torch.tensor([label_map[l.item()] for l in test_labels], dtype=torch.long).to(self.device)


        # Set up Linear Probe
        linear_probe_clf = SSLEvaluator(self.dim_encoder_features, num_classes, 0, 0.0)
        linear_probe_clf.to(self.device)
        _lr = self.lr 
        linear_probe_clf_optimizer = torch.optim.Adam(linear_probe_clf.parameters(), lr=_lr)

        classifier_train_step = 0
        val_step = 0
        best_val_loss = 1e10
        best_val_acc = 0.0
        patience = self.lr_patience
        linear_probe_clf.train()
        best_model = None
        
        # Training loop of the probe
        for e in range(self.probing_epochs):
            train_loss = 0.0
            train_samples = 0.0
            index = 0
            while index + self.mb_size <= len(tr_labels):
                _x = tr_activations[index:index + self.mb_size, :]
                y = tr_labels[index:index + self.mb_size]

                _x = _x.detach()
                # forward pass
                mlp_preds = linear_probe_clf(_x.to(self.device))
                mlp_loss = self.criterion(mlp_preds, y)
                # update finetune weights
                mlp_loss.backward()
                linear_probe_clf_optimizer.step()
                linear_probe_clf_optimizer.zero_grad()
                train_loss += mlp_loss.item()
                train_samples += len(y)


                classifier_train_step += 1
                index += self.mb_size

            # Eval on validation sets
            linear_probe_clf.eval()
            val_loss = 0.0
            acc_correct = 0
            acc_all = 0
            with torch.no_grad():
                singelite = False if len(val_activations) > self.mb_size else True
                index = 0
                while index + self.mb_size < len(val_activations) or singelite:
                    _x = val_activations[index:index + self.mb_size, :]
                    y = val_labels[index:index + self.mb_size]
                    _x = _x.detach();
                    # forward pass
                    mlp_preds = linear_probe_clf(_x.to(self.device))
                    mlp_loss = F.cross_entropy(mlp_preds, y)
                    val_loss += mlp_loss.item()
                    n_corr = (mlp_preds.argmax(1) == y).sum().cpu().item()
                    n_all = y.size()[0]
                    _val_acc = n_corr / n_all
                    acc_correct += n_corr
                    acc_all += n_all
                    val_step += 1
                    index += self.mb_size
                    singelite = False
            
             # mean validation loss
            val_loss = val_loss / acc_all
            val_acc = acc_correct / acc_all

            print(
            f'| Epoch {e} | Train loss: {train_loss:.6f} | Valid loss: {val_loss:.6f} acc: {100 * val_acc:.2f} |'
            )
            
            # Adapt lr
            if val_acc > best_val_acc or best_model is None:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_model = copy.deepcopy(linear_probe_clf.model.state_dict())
                patience = self.lr_patience
                print('*', end='', flush=True)
            else:
                patience -= 1
                if patience <= 0:
                    _lr /= self.lr_factor
                    print(' lr={:.1e}'.format(_lr),)
                    if _lr < self.lr_min:
                        print(' NO MORE PATIENCE')
                        break
                    patience = self.lr_patience
                    linear_probe_clf_optimizer.param_groups[0]['lr'] = _lr
                    linear_probe_clf.model.load_state_dict(best_model)

        linear_probe_clf.model.load_state_dict(best_model)
        linear_probe_clf.eval()

        # Eval on test set
        with torch.no_grad():
            test_loss = 0.0
            acc_correct = 0
            acc_all = 0
            singelite = False if len(test_activations) > self.mb_size else True
            index = 0
            while index + self.mb_size < len(test_activations) or singelite:
                _x = test_activations[index:index + self.mb_size, :]
                y = test_labels[index:index + self.mb_size]
                _x = _x.detach();
                # forward pass
                mlp_preds = linear_probe_clf(_x.to(self.device))
                mlp_loss = F.cross_entropy(mlp_preds, y)
                test_loss += mlp_loss.item()
                n_corr = (mlp_preds.argmax(1) == y).sum().cpu().item()
                n_all = y.size()[0]
                _test_acc = n_corr / n_all
                acc_correct += n_corr
                acc_all += n_all
                index += self.mb_size
                singelite = False
        
            # mean test loss
            test_loss = val_loss / acc_all
            test_acc = acc_correct / acc_all

            print(f'Test loss: {test_loss}, test acc: {test_acc}')

        if self.save_file is not None:
            with open(self.save_file, 'a') as f:
                if val_activations is None:
                    if self.exp_idx is not None:
                        f.write(f'{self.exp_idx},_,{test_acc:.4f}\n')
                    else:
                        f.write(f'_,{test_acc:.4f}\n')
                else:        
                    if self.exp_idx is not None:
                        f.write(f'{self.exp_idx},{best_val_acc:.4f},{test_acc:.4f}\n')
                    else:
                        f.write(f'{best_val_acc:.4f},{test_acc:.4f}\n')



class SSLEvaluator(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super().__init__()
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.out_features = n_classes  # for *head* compability
        if n_hidden is None or n_hidden == 0:
            # use linear classifier
            self.model = nn.Sequential(nn.Flatten(), nn.Dropout(p=p), nn.Linear(n_input, n_classes, bias=True))
        else:
            # use simple MLP classifier
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True),
            )

    def forward(self, x):
        logits = self.model(x)
        return logits