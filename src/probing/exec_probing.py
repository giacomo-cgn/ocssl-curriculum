import os
import torch
from torch.utils.data import ConcatDataset
from typing import List, Dict

from torch.utils.data import ConcatDataset

from . import AbstractProbe
from ..benchmark import Benchmark 
from .analyze_collpase import analyze_collapse


def exec_probing(kwargs: Dict,
                 probes: List[AbstractProbe],
                 probing_benchmark: Benchmark,
                 encoder: torch.nn,
                 pretr_exp_idx: int,
                 probing_tr_ratio_arr: List[float],
                 save_pth: str,
                 ):
    
    for probe in probes:
        probe_save_pth = probe_save_pth = os.path.join(save_pth, f'probe_{probe.get_name()}')
        print(f'==== Probe {probe.get_name()} ==== ')
                
        # PROBING JOINT
        probe_joint_dataset_tr = ConcatDataset(probing_benchmark.train_stream)
        probe_joint_dataset_test = ConcatDataset(probing_benchmark.test_stream)
        if kwargs['val_ratio'] > 0:
            probe_joint_dataset_val = ConcatDataset(probing_benchmark.valid_stream)

        for probing_tr_ratio in probing_tr_ratio_arr:
            # Create probing accuracy log file
            pth = os.path.join(probe_save_pth, f'probing_joint/probing_ratio{probing_tr_ratio}')
            if not os.path.exists(pth):
                os.makedirs(pth)
            probe_save_file = os.path.join(pth, f'probe_exp_{pretr_exp_idx}.csv')
                                        
            print(f'-- Joint Probing, probe tr ratio: {probing_tr_ratio} --')

            if kwargs['val_ratio'] > 0:
                probe.probe(encoder=encoder, tr_dataset=probe_joint_dataset_tr, test_dataset=probe_joint_dataset_test,
                            val_dataset=probe_joint_dataset_val, tr_samples_ratio=probing_tr_ratio,
                            save_file=probe_save_file, dataset_name=probing_benchmark.dataset_name)
            else:
                probe.probe(encoder=encoder, tr_dataset=probe_joint_dataset_tr, test_dataset=probe_joint_dataset_test,
                            tr_samples_ratio=probing_tr_ratio, save_file=probe_save_file, dataset_name=probing_benchmark.dataset_name)
               
    if kwargs["analyze_collapse"]:
        collapse_pth = os.path.join(save_pth, 'collapse', f'pretr_exp_{pretr_exp_idx}')
        if not os.path.exists(collapse_pth):
            os.makedirs(collapse_pth)
        if kwargs['val_ratio'] > 0:
            analyze_collapse(encoder=encoder, test_dataset=probe_joint_dataset_test, val_dataset=probe_joint_dataset_val,
                             mb_size=kwargs["eval_mb_size"], device=probe.device, save_path=collapse_pth, dataset_name=probing_benchmark.dataset_name)