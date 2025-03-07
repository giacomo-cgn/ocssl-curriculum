import os
import torch
from torch.utils.data import ConcatDataset
from typing import List, Dict

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.dataset import Dataset
from torch import nn

from ..utils import SupervisedDataset
from . import AbstractProbe
from ..benchmark import Benchmark 
from .analyze_collpase import analyze_collapse
from .prototype_drift import prototype_drift


def exec_probing(kwargs: Dict,
                 probes: List[AbstractProbe],
                 probing_benchmark: Benchmark,
                 encoder: torch.nn,
                 pretr_exp_idx: int,
                 save_pth: str,
                 device: str,
                 prev_classes: List[int] = None,
                 curr_classes: List[int] = None,
                 ):
    
    probe_joint_dataset_tr = ConcatDataset(probing_benchmark.train_stream)
    probe_joint_dataset_test = ConcatDataset(probing_benchmark.test_stream)
    if kwargs['val_ratio'] > 0:
        probe_joint_dataset_val = ConcatDataset(probing_benchmark.valid_stream)
    else:
        probe_joint_dataset_val = None
    
    tr_activations, tr_labels, val_activations, val_labels, test_activations, test_labels = extract_representations(
        encoder=encoder,
        tr_dataset=probe_joint_dataset_tr,
        val_dataset=probe_joint_dataset_val,
        test_dataset=probe_joint_dataset_test,
        dataset_name=probing_benchmark.dataset_name,
        device=device,
        mb_size=kwargs['eval_mb_size'],
    )
    
    for probe in probes:
        probe_save_pth = os.path.join(save_pth, f'probe_{probe.get_name()}', 'probing_joint')
        if not os.path.exists(probe_save_pth):
            os.makedirs(probe_save_pth)
        probe_save_file = os.path.join(probe_save_pth, f'probe_exp_{pretr_exp_idx}.csv')

        print(f'==== Probe {probe.get_name()} ==== ')

        probe.probe(tr_activations=tr_activations, tr_labels=tr_labels, val_activations=val_activations,
                    val_labels=val_labels, test_activations=test_activations, test_labels=test_labels,
                    save_file=probe_save_file)

               
    if kwargs["analyze_collapse"]:
        print(f'==== Analyzing Collapse ==== ')
        collapse_pth = os.path.join(save_pth, 'collapse', f'pretr_exp_{pretr_exp_idx}')
        if not os.path.exists(collapse_pth):
            os.makedirs(collapse_pth)
        analyze_collapse(tr_activations=tr_activations, tr_labels=tr_labels, val_activations=val_activations,
            val_labels=val_labels, test_activations=test_activations, test_labels=test_labels, save_path=collapse_pth)
            
    
    if kwargs['analyze_drift']:
        print(f'==== Analyzing Drift ==== ')
        drift_pth = os.path.join(save_pth, 'drift')

        prototype_drift(tr_activations=tr_activations, tr_labels=tr_labels, val_activations=val_activations,
                        val_labels=val_labels, test_activations=test_activations, test_labels=test_labels,
                        save_path=drift_pth, pretr_exp_idx=pretr_exp_idx, previous_classes=prev_classes, current_classes=curr_classes)

            


# Extract representations of the training set and test set
def extract_representations(encoder: nn,
                            tr_dataset: Dataset,
                            test_dataset: Dataset,
                            val_dataset: Dataset = None,
                            dataset_name: str = 'cifar100',
                            device: str = 'cpu',
                            mb_size: int = 512,
                            ):
    tr_dataset = SupervisedDataset(tr_dataset, dataset_name)
    train_loader = DataLoader(dataset=tr_dataset, batch_size=mb_size, shuffle=True, num_workers=8)
    test_dataset = SupervisedDataset(test_dataset, dataset_name)
    test_loader = DataLoader(dataset=test_dataset, batch_size=mb_size, shuffle=False, num_workers=8)
    if val_dataset is not None:
        val_dataset = SupervisedDataset(val_dataset, dataset_name)
        val_loader = DataLoader(dataset=val_dataset, batch_size=mb_size, shuffle=False, num_workers=8)


    print('EVALUATION: Extracting representations...')
    with torch.no_grad():
        # Put encoder in eval mode, as even with no gradient it could interfere with batchnorm
        encoder.eval()

        # Get encoder activations for tr dataloader
        tr_activations_list = []
        tr_labels_list = []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            activations = encoder(inputs)
            tr_activations_list.append(activations.detach())
            tr_labels_list.append(labels.detach())
        tr_activations = torch.cat(tr_activations_list, dim=0)
        tr_labels = torch.cat(tr_labels_list, dim=0)

        if val_dataset is not None:
            # Get encoder activations for val dataloader
            val_activations_list = []
            val_labels_list = []
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                activations = encoder(inputs)
                val_activations_list.append(activations.detach())
                val_labels_list.append(labels.detach())
            val_activations = torch.cat(val_activations_list, dim=0)
            val_labels = torch.cat(val_labels_list, dim=0)
        else:
            val_activations = None
            val_labels = None

        # Get encoder activations for test dataloader
        test_activations_list = []
        test_labels_list = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            activations = encoder(inputs)
            test_activations_list.append(activations.detach())
            test_labels_list.append(labels.detach())
        test_activations = torch.cat(test_activations_list, dim=0)
        test_labels = torch.cat(test_labels_list, dim=0)

    return tr_activations, tr_labels, val_activations, val_labels, test_activations, test_labels