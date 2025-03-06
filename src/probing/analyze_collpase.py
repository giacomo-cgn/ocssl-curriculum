import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ..utils import SupervisedDataset


def analyze_collapse(encoder: nn,
                     test_dataset: Dataset,
                     val_dataset: Dataset = None,
                     mb_size: int = 256,
                     device: str = 'cpu',
                     save_path: str = None,
                     dataset_name: str = 'cifar100',
              ):
            
    test_dataset = SupervisedDataset(test_dataset, dataset_name)
    test_loader = DataLoader(dataset=test_dataset, batch_size=mb_size, shuffle=False, num_workers=8)
    if val_dataset is not None:
        val_dataset = SupervisedDataset(val_dataset, dataset_name)
        val_loader = DataLoader(dataset=val_dataset, batch_size=mb_size, shuffle=False, num_workers=8)

    print('=== Analyzing collapse ... ===')

    with torch.no_grad():
        # Put encoder in eval mode, as even with no gradient it could interfere with batchnorm
        encoder.eval()    # Get encoder activations for val dataloader

        if val_dataset is not None:
            val_activations_list = []
            val_labels_list = []
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                activations = encoder(inputs)
                val_activations_list.append(activations.detach())
                val_labels_list.append(labels.detach())
            val_activations = torch.cat(val_activations_list, dim=0).cpu()
            val_labels = torch.cat(val_labels_list, dim=0).cpu()

        # Get encoder activations for test dataloader
        test_activations_list = []
        test_labels_list = []
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            activations = encoder(inputs)
            test_activations_list.append(activations.detach())
            test_labels_list.append(labels.detach())
        test_activations = torch.cat(test_activations_list, dim=0).cpu()
        test_labels = torch.cat(test_labels_list, dim=0).cpu()

        # ------------- Class agnostic collapse -----------------
        test_svd = get_svd(test_activations)
        if val_dataset is not None:
            val_svd = get_svd(val_activations)

        plt.figure(figsize=(15,5))
    
        plt.subplot(1,3,1)
        plt.plot(test_svd, label='Test')
        if val_dataset is not None:
            plt.plot(val_svd, label='Validation')
        plt.legend()
        plt.xlabel("Sorted singular value index")
        plt.ylabel('Singular Value')
        plt.yscale('log')

        plt.subplot(1,3,2)
        plt.plot(test_svd/test_svd[0], label='Test')
        if val_dataset is not None:
            plt.plot(val_svd/val_svd[0], label='Validation')
        plt.legend()
        plt.xlabel("Sorted singular value index")
        plt.ylabel('Normalized Singular Value')
        plt.yscale('log')

        plt.subplot(1,3,3)
        plt.plot(np.cumsum(test_svd) / test_svd.sum(), label='Test')
        if val_dataset is not None:
            plt.plot(np.cumsum(val_svd) / val_svd.sum(),  label='Validation')
        plt.legend()
        plt.xlabel("Sorted singular value index")
        plt.ylabel('Cumulative Explained Variance')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'class_agnostic_collapse.png'), dpi=300)
        plt.cla()

        with open(os.path.join(save_path, 'class_agnostic_auc.txt'), 'a') as f:
            f.write(f'Test AUC: {auc(test_svd):.4f}\n')
            if val_dataset is not None:
                f.write(f'Validation AUC: {auc(val_svd):.4f}\n')

        # --------- Class specific collapse -------------

        class_test_svd = []
        class_val_svd = []
        for label in range(np.abs(test_labels).max()+1):
            test_svd = get_svd(test_activations[test_labels == label])
            class_test_svd.append(test_svd)
            if val_dataset is not None:
                val_svd = get_svd(val_activations[val_labels == label])
                class_val_svd.append(val_svd)

        plt.figure(figsize=(15, 10))

        plt.subplot(2,3,1)
        for label, svd in enumerate(class_test_svd):
            plt.plot(svd, label=f'Class {label}')
        plt.xlabel("Sorted singular value index")
        plt.xlabel("Sorted singular value index")
        plt.ylabel('Singular Value')
        plt.yscale('log')
        plt.title('Test')

        plt.subplot(2,3,2)
        for label, svd in enumerate(class_test_svd):
            plt.plot(svd/svd[0], label=f'Class {label}')
        plt.xlabel("Sorted singular value index")
        plt.ylabel('Normalized Singular Value')
        plt.yscale('log')
        plt.title('Test')

        plt.subplot(2,3,3)
        for label, svd in enumerate(class_test_svd):
            plt.plot(np.cumsum(svd)/svd.sum(), label=f'Class {label}')
        plt.xlabel("Sorted singular value index")
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Test')

        if len(class_val_svd) > 0:
            plt.subplot(2,3,4)
            for label, svd in enumerate(class_val_svd):
                plt.plot(svd, label=f'Class {label}')
            plt.xlabel("Sorted singular value index")
            plt.xlabel("Sorted singular value index")
            plt.ylabel('Singular Value')
            plt.yscale('log')
            plt.title('Validation')

            plt.subplot(2,3,5)
            for label, svd in enumerate(class_val_svd):
                plt.plot(svd/svd[0], label=f'Class {label}')
            plt.xlabel("Sorted singular value index")
            plt.ylabel('Normalized Singular Value')
            plt.yscale('log')
            plt.title('Validation')

            plt.subplot(2,3,6)
            for label, svd in enumerate(class_val_svd):
                 plt.plot(np.cumsum(svd)/svd.sum(), label=f'Class {label}')
            plt.xlabel("Sorted singular value index")
            plt.ylabel('Cumulative Explained Variance')
            plt.title('Validation')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'class_specific_collapse.png'), dpi=300)
        plt.cla()

        with open(os.path.join(save_path, 'class_specific_auc.csv'), 'a') as f:
            if val_dataset is not None:
                f.write(f'class,val_auc,test_auc\n')
                for label, (val_svd, test_svd) in enumerate(zip(class_val_svd, class_test_svd)):
                    f.write(f'{label},{auc(val_svd):.4f},{auc(test_svd):.4f}\n')
            else:
                f.write(f'class,test_auc\n')
                for label, test_svd in enumerate(class_test_svd):
                    f.write(f'{label},{auc(test_svd):.4f}\n')




def get_svd(activations):
    reprs = activations.reshape(-1, activations.shape[-1])

    norms = torch.linalg.norm(reprs, dim=1)
    normed_reprs = reprs / (1e-6 + norms.unsqueeze(1))
    svd = torch.svd(normed_reprs).S
    return svd

def auc(singular_values):
    # Equation 2 from https://arxiv.org/abs/2209.15007
    explvar = np.cumsum(singular_values) / singular_values.sum()
    return explvar.sum() / len(explvar)