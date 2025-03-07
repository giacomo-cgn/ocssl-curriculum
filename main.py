import torch

import os
import datetime
import tqdm as tqdm
import numpy as np
import pandas as pd

from src.get_curriculum import get_curriculum
from src.probing import exec_probing, ProbingSklearn, ProbingPytorch
from src.backbones import get_encoder

from src.ssl_models import BarlowTwins, SimSiam, BYOL, MoCo, SimCLR, EMP, MAE, SimSiamMultiview, BYOLMultiview, recover_ssl_model

from src.strategies import NoStrategy, Replay, ARP, AEP, APRE, LUMP, MinRed, CaSSLe, CaSSLeR, ReplayEMP
from src.standalone_strategies import OsirisR

from src.trainer import Trainer

from src.buffers import get_buffer

from src.utils import write_final_scores, read_command_line_args, save_avg_stream_acc

from src.get_datasets import get_benchmark

import time

def exec_experiment(**kwargs):
    buffer_free_strategies = ['no_strategy', 'aep', 'cassle']

    # Set up save folders
    str_now = datetime.datetime.now().strftime("%d-%m-%y_%H:%M")

    if kwargs['no_train']:
        folder_name = f'notrain_{kwargs["dataset"]}_{str_now}'
    else:
        f'{kwargs["name"]}_{kwargs["strategy"]}_{kwargs["model"]}_{kwargs["dataset"]}_{str_now}'
        folder_name = f'{kwargs["strategy"]}_{kwargs["model"]}_{kwargs["dataset"]}_{str_now}'
    save_pth = os.path.join(kwargs["save_folder"], f'{folder_name}_{kwargs["name"]}')
    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    

    # Save general kwargs
    with open(save_pth + '/config.txt', 'a') as f:
        f.write('\n')
        f.write(f'---- EXPERIMENT CONFIGS ----\n')
        f.write(f'Seed: {kwargs["seed"]}\n')
        f.write(f'Dataset Seed: {kwargs["dataset_seed"]}\n')
        f.write(f'Experiment Date: {str_now}\n')
        f.write(f'Model: {kwargs["model"]}\n')
        f.write(f'Encoder: {kwargs["encoder"]}\n')
        f.write(f'Dataset: {kwargs["dataset"]}\n')
        f.write(f'MB Passes: {kwargs["mb_passes"]}\n')
        f.write(f'Tot Training Steps: {kwargs["tot_tr_steps"]}\n')
        f.write(f'Online transforms: {kwargs["online_transforms"]}\n')
        
        f.write(f'Memory Size: {kwargs["mem_size"]}\n')
        f.write(f'Train MB Size: {kwargs["tr_mb_size"]}\n')
        f.write(f'Replay MB Size: {kwargs["repl_mb_size"]}\n')
        f.write(f'Save final model: {kwargs["save_model_final"]}\n')
        f.write(f'-- Pretrained weights initialization configs --\n')
        f.write(f'Pretrain init: {kwargs["pretrain_init_type"]}\n')
        if kwargs["pretrain_init_type"] == 'encoder' or kwargs["pretrain_init_type"] == 'ssl':
            f.write(f'Pretrain init source: {kwargs["pretrain_init_source"]}\n')
            f.write(f'Pretrain init path: {kwargs["pretrain_init_pth"]}\n')

        f.write(f'-- Probing configs --\n')
        f.write(f'Validation Ratio: {kwargs["val_ratio"]}\n')

        f.write(f'-- Curriculum configs --\n')
        f.write(f'total training steps: {kwargs["tot_tr_steps"]}\n')
        f.write(f'Intermediate evaluation: {kwargs["intermediate_eval"]}\n')
        f.write(f'Eval every steps: {kwargs["eval_every_steps"]}\n')
        f.write(f'Curriculum: {kwargs["curriculum"]}\n')


    curriculum, image_size = get_curriculum(
        curriculum_order_list=kwargs["curriculum"],
        dataset_name=kwargs["dataset"],
        total_steps=kwargs["tot_tr_steps"],
        kwargs=kwargs
        )

    # Set seed (After get_benchmark!)
    torch.manual_seed(kwargs["seed"])
    np.random.default_rng(kwargs["seed"])

    # Device
    if torch.cuda.is_available():       
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        if kwargs["gpu_idx"] < torch.cuda.device_count():
            device = torch.device(f"cuda:{kwargs['gpu_idx']}")
        else:
            device = torch.device("cuda")
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    # Encoder
    encoder, dim_encoder_features = get_encoder(encoder_name=kwargs["encoder"],
                                                image_size=image_size,
                                                ssl_model_name=kwargs["model"],
                                                vit_avg_pooling=kwargs["vit_avg_pooling"],
                                                pretrain_init_type=kwargs["pretrain_init_type"],
                                                pretrain_init_source=kwargs["pretrain_init_source"],
                                                pretrain_init_pth=kwargs["pretrain_init_pth"],
                                                save_pth=save_pth
                                                )
    


    # Buffer
    if not kwargs["strategy"] in buffer_free_strategies:
        assert kwargs["online_transforms"] == True, "Buffer-based strategies require online_transforms=True"
        if kwargs["buffer_type"] == "default":
            # Set default buffer for each strategy
            if kwargs["strategy"] in ['replay', 'apre', 'arp', 'lump', 'double_resnet', 'osiris_r', 'cassle_r']:
                kwargs["buffer_type"] = "reservoir"
            elif kwargs["strategy"] == "minred":
                kwargs["buffer_type"] = "minred"
            elif kwargs["strategy"] == "replay_emp":
                kwargs["buffer_type"] = "aug_rep"
            elif kwargs["strategy"] == "arp_hybrid":
                kwargs["buffer_type"] = "hybrid_minred_fifo"
            else:
                raise Exception(f'Strategy {kwargs["strategy"]} not supported for default buffer')
        # Enforce buffer constraints for certain strategies  
        elif kwargs["buffer_type"] == "aug_rep" and not kwargs["strategy"] == "replay_emp":
            raise Exception(f"Buffer type {kwargs['buffer_type']} is only compatible with strategy 'replay_emp'")
        
        buffer = get_buffer(buffer_type=kwargs["buffer_type"], mem_size=kwargs["mem_size"],
                            alpha_ema=kwargs["features_buffer_ema"], fifo_buffer_ratio=kwargs["fifo_buffer_ratio"],
                            device=device)

        # Save buffer configs
        with open(save_pth + '/config.txt', 'a') as f:
            f.write('\n')
            f.write(f'---- BUFFER CONFIGS ----\n')
            f.write(f'Buffer Type: {kwargs["buffer_type"]}\n')
            f.write(f'Buffer Size: {kwargs["mem_size"]}\n')
            if kwargs["buffer_type"] in ["minred", "reservoir", "fifo"]:
                f.write(f'Features update EMA param (MinRed): {kwargs["features_buffer_ema"]}\n')
            if kwargs["buffer_type"] in ['hybrid_minred_fifo']:
                f.write(f'FIFO Buffer Ratio: {kwargs["fifo_buffer_ratio"]}\n')


    if kwargs["aligner_dim"] <= 0:
        aligner_dim = kwargs["dim_pred"]
    else:
        aligner_dim = kwargs["aligner_dim"]
    
    # ---- SSL model ----
    if kwargs["model"] == 'simsiam':
        ssl_model = SimSiam(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                            dim_proj=kwargs["dim_proj"], dim_pred=kwargs["dim_pred"],
                            save_pth=save_pth)
        num_views = 2
    elif kwargs["model"] == 'simsiam_multiview':
        ssl_model = SimSiamMultiview(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                                        dim_proj=kwargs["dim_proj"], dim_pred=kwargs["dim_pred"],
                                        n_patches=kwargs["num_views"], save_pth=save_pth)
        num_views = kwargs["num_views"]

    elif kwargs["model"] == 'byol':
        ssl_model = BYOL(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                            dim_proj=kwargs["dim_proj"], dim_pred=kwargs["dim_pred"],
                            byol_momentum=kwargs["byol_momentum"], return_momentum_encoder=kwargs["return_momentum_encoder"],
                            save_pth=save_pth)
        num_views = 2
    elif kwargs["model"] == 'byol_multiview':
        ssl_model = BYOLMultiview(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                                    dim_proj=kwargs["dim_proj"], dim_pred=kwargs["dim_pred"],
                                    byol_momentum=kwargs["byol_momentum"], return_momentum_encoder=kwargs["return_momentum_encoder"],
                                    n_patches=kwargs["num_views"], save_pth=save_pth)
        num_views = kwargs["num_views"]
        
    elif kwargs["model"] == 'barlow_twins':
        ssl_model = BarlowTwins(encoder=encoder, dim_backbone_features=dim_encoder_features,
                                dim_features=kwargs["dim_proj"],
                                lambd=kwargs["lambd"], save_pth=save_pth)
        num_views = 2

    elif kwargs["model"] == 'moco':
        ssl_model = MoCo(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                            dim_proj=kwargs["dim_proj"],
                            moco_momentum=kwargs["moco_momentum"], moco_queue_size=kwargs["moco_queue_size"],
                            moco_temp=kwargs["moco_temp"],return_momentum_encoder=kwargs["return_momentum_encoder"],
                            queue_type=kwargs["moco_queue_type"],
                            save_pth=save_pth, device=device)
        num_views = 2

    elif kwargs["model"] == 'simclr':
        ssl_model = SimCLR(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                            dim_proj=kwargs["dim_proj"], temperature=kwargs["simclr_temp"],
                            save_pth=save_pth)
        num_views = 2

    elif kwargs["model"] == 'emp':
        ssl_model = EMP(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                        dim_proj=kwargs["dim_proj"], n_patches=kwargs["num_views"],
                        emp_tcr_param=kwargs["emp_tcr_param"], emp_tcr_eps=kwargs["emp_tcr_eps"], 
                        emp_patch_sim=kwargs["emp_patch_sim"], save_pth=save_pth)
        num_views = kwargs["num_views"]

    elif kwargs["model"] == 'osiris_r':
        ssl_model = OsirisR(base_encoder=encoder, dim_backbone_features=dim_encoder_features,
                                dim_proj=kwargs["dim_proj"], buffer=buffer, device=device,
                                replay_mb_size=kwargs["repl_mb_size"],
                                save_pth=save_pth)
        num_views = 2
        assert kwargs["strategy"] == kwargs["model"], 'Strategy and SSL model must be the same for Osiris-R'


    elif kwargs["model"] == 'mae':
        ssl_model = MAE(vit_encoder=encoder,
                        image_size=image_size, patch_size=kwargs["mae_patch_size"], emb_dim=kwargs["mae_emb_dim"],
                        decoder_layer=kwargs["mae_decoder_layer"], decoder_head=kwargs["mae_decoder_head"],
                        mask_ratio=kwargs["mae_mask_ratio"], save_pth=save_pth)
        num_views = 1
        
    else:
        raise Exception(f'Invalid model {kwargs["model"]}') 
    
    # Initialization from pretrained weights of SSL model
    if kwargs["pretrain_init_type"] == 'ssl':
        if kwargs["pretrain_init_source"] == 'path':
            ssl_model = recover_ssl_model(ssl_model, kwargs["pretrain_init_pth"])
        else:
            raise Exception(f'Invalid pretrain_init_source for ssl type pretrain initialization: {kwargs["pretrain_init_source"]}')
        
    ssl_model = ssl_model.to(device)
            
    
    # ---- Strategy ----
    if kwargs["strategy"] == 'no_strategy':
        strategy = NoStrategy(ssl_model=ssl_model, device=device, save_pth=save_pth)

    elif kwargs["strategy"] == 'replay':
        strategy = Replay(ssl_model=ssl_model, device=device, save_pth=save_pth,
                        buffer=buffer, replay_mb_size=kwargs["repl_mb_size"])
        
    elif kwargs["strategy"] == 'arp':
        strategy = ARP(ssl_model=ssl_model, device=device, save_pth=save_pth,
                    buffer=buffer, replay_mb_size=kwargs["repl_mb_size"],
                    omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                    use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"], 
                    aligner_dim=aligner_dim)
    
    elif kwargs["strategy"] == 'aep':
        strategy = AEP(ssl_model=ssl_model, device=device, save_pth=save_pth,
                    omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                    use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"], 
                    aligner_dim=aligner_dim, momentum_ema=kwargs["momentum_ema"])
    
    elif kwargs["strategy"] == 'apre':
        strategy = APRE(ssl_model=ssl_model, device=device, save_pth=save_pth,
                        buffer=buffer, replay_mb_size=kwargs["repl_mb_size"],
                        omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                        use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"], 
                        aligner_dim=aligner_dim, momentum_ema=kwargs["momentum_ema"])
        
    elif kwargs["strategy"] == 'lump':
        strategy = LUMP(ssl_model=ssl_model, device=device, save_pth=save_pth,
                        buffer=buffer,
                        alpha_lump=kwargs["alpha_lump"])
        
    elif kwargs["strategy"] == 'minred':
        strategy = MinRed(ssl_model=ssl_model, device=device, save_pth=save_pth,
                        buffer=buffer, replay_mb_size=kwargs["repl_mb_size"])
    
    elif kwargs["strategy"] == 'cassle':
        strategy = CaSSLe(ssl_model=ssl_model, device=device, save_pth=save_pth,
                        omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                        use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"], 
                        aligner_dim=aligner_dim)
        
    elif kwargs["strategy"] == 'cassle_r':
        strategy = CaSSLeR(ssl_model=ssl_model, device=device, save_pth=save_pth,
                        buffer=buffer, replay_mb_size=kwargs["repl_mb_size"],
                        omega=kwargs["omega"], align_criterion=kwargs["align_criterion"],
                        use_aligner=kwargs["use_aligner"], align_after_proj=kwargs["align_after_proj"], 
                        aligner_dim=aligner_dim)

    elif kwargs["strategy"] == 'osiris_r':
        strategy = ssl_model # SSL model and strategy are combined
        
    elif kwargs["strategy"] == 'replay_emp':
        assert kwargs["buffer_type"] == "aug_rep", "Buffer type must be 'aug_rep_buffer' (AugmentedRepresentationsBuffer) for 'replay_emp' strategy"
        assert kwargs["model"] == 'emp', "SSL model has to be 'emp' for 'replay_emp' strategy"
        strategy = ReplayEMP(ssl_model=ssl_model, device=device, save_pth=save_pth,
                            buffer=buffer, replay_mb_size=kwargs["repl_mb_size"],
                            emp_loss=ssl_model.get_criterion()[0], emp_tcr_param=kwargs["emp_tcr_param"],
                            emp_tcr_eps=kwargs["emp_tcr_eps"], emp_patch_sim=kwargs["emp_patch_sim"])

    else:
        raise Exception(f'Strategy {kwargs["strategy"]} not supported')

    # Set up the trainer wrapper
    trainer = Trainer(ssl_model=ssl_model, strategy=strategy, optim=kwargs["optim"], lr=kwargs["lr"], momentum=kwargs["optim_momentum"],
                        lars_eta= kwargs["lars_eta"],
                        weight_decay=kwargs["weight_decay"], train_mb_size=kwargs["tr_mb_size"],
                        device=device, dataset_name=kwargs["dataset"], save_pth=save_pth,
                        save_model=kwargs["save_model_every_exp"], online_transforms=kwargs["online_transforms"], num_views=num_views)

    # Init probing
    
    probes = []
    if kwargs["probing_rr"]:
         probes.append(ProbingSklearn(probe_type='rr', device=device, mb_size=kwargs["eval_mb_size"],
                               seed=kwargs["seed"], config_save_pth=save_pth))
    if kwargs["probing_knn"]:
         probes.append(ProbingSklearn(probe_type='knn', device=device, mb_size=kwargs["eval_mb_size"],
                               knn_k=kwargs["knn_k"], seed=kwargs["seed"], config_save_pth=save_pth))
         
    if kwargs["probing_torch"]:
        probes.append(ProbingPytorch(device=device, mb_size=kwargs["eval_mb_size"], config_save_pth=save_pth,
                                 dim_encoder_features=dim_encoder_features, lr=kwargs["probe_lr"],
                                 lr_patience=kwargs["probe_lr_patience"], lr_factor=kwargs["probe_lr_factor"],
                                 lr_min=kwargs["probe_lr_min"], probing_epochs=kwargs["probe_epochs"]))
       

    
    probing_benchmark, image_size = get_benchmark(kwargs["dataset"], kwargs["dataset_root"], 20, kwargs["dataset_seed"], val_ratio=0.1)

    training_time_tot = 0 

    if kwargs["no_train"]:
        # No SSL training is done, only using the randomly initialized encoder as feature extractor
        exec_probing(kwargs=kwargs, probes=probes, probing_benchmark=probing_benchmark, encoder=encoder, pretr_exp_idx=0,
                     save_pth=save_pth, device=device)

    else:
        # Self supervised training over the experiences
        before_tr_steps = 0
        eval_idx = 0

        training_time_start = time.time()
        for exp_idx, curriculum_part in enumerate(curriculum):
            print(f'==== Beginning self supervised training for experience: {exp_idx} ====')

            if kwargs["intermediate_eval"]:
                # Evaluate iid trained model during training (not only at the end)
                intermediate_eval_dict = {
                    'status': True,
                    'kwargs': kwargs,
                    'probes': probes,
                    'benchmark': probing_benchmark,
                }
            else:
                intermediate_eval_dict = {
                    'status': False,
                }


            trained_ssl_model, eval_idx = trainer.train_experience(curriculum_part.dataset, curriculum_part.tr_steps, exp_idx,
                                                                   before_tr_steps, kwargs["eval_every_steps"], eval_idx,
                                                                   intermediate_eval_dict)
            training_time_tot += time.time() - training_time_start
        
        if not kwargs["intermediate_eval"]:
            # Probe only at the end of training
            exec_probing(kwargs=kwargs, probes=probes, probing_benchmark=probing_benchmark, encoder=trained_ssl_model.get_encoder_for_eval(), 
                     pretr_exp_idx=eval_idx, save_pth=save_pth, device=device)
            
    # Save training time
            with open(os.path.join(save_pth, 'training_time.txt'), 'w') as f:
                f.write(f'Total training time: {training_time_tot} seconds')
                
        
    # Calculate and save final probing scores
    for probe in probes:
        probe_pth = os.path.join(save_pth, f'probe_{probe.get_name()}')
        
        write_final_scores(probe=probe.get_name(), folder_input_path=os.path.join(probe_pth, 'probing_joint'),
                        output_file=os.path.join(save_pth, 'final_scores_joint.csv'))
        if kwargs["intermediate_eval"]:
            save_avg_stream_acc(probe=probe.get_name(), save_pth=save_pth)

        
    # Save final pretrained model
    if kwargs["save_model_final"]:
        chkpt_pth = os.path.join(save_pth, 'checkpoints')
        if not os.path.exists(chkpt_pth):
            os.makedirs(chkpt_pth)
        if kwargs["no_train"]:
            torch.save(encoder.state_dict(),
                    os.path.join(chkpt_pth, f'final_model_state.pth'))
        else:
            torch.save(trained_ssl_model.state_dict(),
                    os.path.join(chkpt_pth, f'final_model_state.pth'))


    return save_pth





if __name__ == '__main__':
    # Parse arguments
    args = read_command_line_args()

    exec_experiment(**args.__dict__)
