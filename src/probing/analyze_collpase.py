import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def analyze_collapse(tr_activations: torch.Tensor,
                     tr_labels: torch.Tensor,
                     test_activations: torch.Tensor,
                     test_labels: torch.Tensor,
                     val_activations: torch.Tensor = None,
                     val_labels: torch.Tensor = None,
                     save_path: str = None,
              ):
    

    tr_activations = tr_activations.cpu()
    tr_labels = tr_labels.cpu()
    test_activations = test_activations.cpu()
    test_labels = test_labels.cpu()
    if val_activations is not None:
        val_activations = val_activations.cpu()
        val_labels = val_labels.cpu()

        # ------------- Class agnostic collapse -----------------
        tr_svd = get_svd(tr_activations)
        test_svd = get_svd(test_activations)
        if val_activations is not None:
            val_svd = get_svd(val_activations)

        plt.figure(figsize=(15,5))
    
        plt.subplot(1,3,1)
        plt.plot(tr_svd, label='Train')
        plt.plot(test_svd, label='Test')
        if val_activations is not None:
            plt.plot(val_svd, label='Validation')
        plt.legend()
        plt.xlabel("Sorted singular value index")
        plt.ylabel('Singular Value')
        plt.yscale('log')

        plt.subplot(1,3,2)
        plt.plot(tr_svd/tr_svd[0], label='Train')
        plt.plot(test_svd/test_svd[0], label='Test')
        if val_activations is not None:
            plt.plot(val_svd/val_svd[0], label='Validation')
        plt.legend()
        plt.xlabel("Sorted singular value index")
        plt.ylabel('Normalized Singular Value')
        plt.yscale('log')

        plt.subplot(1,3,3)
        plt.plot(np.cumsum(tr_svd) / tr_svd.sum(), label='Train')
        plt.plot(np.cumsum(test_svd) / test_svd.sum(), label='Test')
        if val_activations is not None:
            plt.plot(np.cumsum(val_svd) / val_svd.sum(),  label='Validation')
        plt.legend()
        plt.xlabel("Sorted singular value index")
        plt.ylabel('Cumulative Explained Variance')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'class_agnostic_collapse.png'), dpi=300)
        plt.cla()

        with open(os.path.join(save_path, 'class_agnostic_auc.txt'), 'a') as f:
            f.write(f'Train AUC: {auc(tr_svd):.4f}\n')
            if val_activations is not None:
                f.write(f'Validation AUC: {auc(val_svd):.4f}\n')
            f.write(f'Test AUC: {auc(test_svd):.4f}\n')

        # --------- Class specific collapse -------------

        class_tr_svd = []
        class_test_svd = []
        class_val_svd = []
        for label in range(np.abs(test_labels).max()+1):
            test_svd = get_svd(test_activations[test_labels == label])
            class_test_svd.append(test_svd)
            tr_svd = get_svd(tr_activations[tr_labels == label])
            class_tr_svd.append(tr_svd)
            if val_activations is not None:
                val_svd = get_svd(val_activations[val_labels == label])
                class_val_svd.append(val_svd)

        plt.figure(figsize=(15, 15))

        plt.subplot(3,3,1)
        for label, svd in enumerate(class_tr_svd):
            plt.plot(svd, label=f'Class {label}')
        plt.xlabel("Sorted singular value index")
        plt.xlabel("Sorted singular value index")
        plt.ylabel('Singular Value')
        plt.yscale('log')
        plt.title('Train')

        plt.subplot(3,3,2)
        for label, svd in enumerate(class_tr_svd):
            plt.plot(svd/svd[0], label=f'Class {label}')
        plt.xlabel("Sorted singular value index")
        plt.ylabel('Normalized Singular Value')
        plt.yscale('log')
        plt.title('Train')

        plt.subplot(3,3,3)
        for label, svd in enumerate(class_tr_svd):
            plt.plot(np.cumsum(svd)/svd.sum(), label=f'Class {label}')
        plt.xlabel("Sorted singular value index")
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Train')

        plt.subplot(3,3,4)
        for label, svd in enumerate(class_test_svd):
            plt.plot(svd, label=f'Class {label}')
        plt.xlabel("Sorted singular value index")
        plt.xlabel("Sorted singular value index")
        plt.ylabel('Singular Value')
        plt.yscale('log')
        plt.title('Test')

        plt.subplot(3,3,5)
        for label, svd in enumerate(class_test_svd):
            plt.plot(svd/svd[0], label=f'Class {label}')
        plt.xlabel("Sorted singular value index")
        plt.ylabel('Normalized Singular Value')
        plt.yscale('log')
        plt.title('Test')

        plt.subplot(3,3,6)
        for label, svd in enumerate(class_test_svd):
            plt.plot(np.cumsum(svd)/svd.sum(), label=f'Class {label}')
        plt.xlabel("Sorted singular value index")
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Test')

        if len(class_val_svd) > 0:
            plt.subplot(3,3,7)
            for label, svd in enumerate(class_val_svd):
                plt.plot(svd, label=f'Class {label}')
            plt.xlabel("Sorted singular value index")
            plt.xlabel("Sorted singular value index")
            plt.ylabel('Singular Value')
            plt.yscale('log')
            plt.title('Validation')

            plt.subplot(3,3,8)
            for label, svd in enumerate(class_val_svd):
                plt.plot(svd/svd[0], label=f'Class {label}')
            plt.xlabel("Sorted singular value index")
            plt.ylabel('Normalized Singular Value')
            plt.yscale('log')
            plt.title('Validation')

            plt.subplot(3,3,9)
            for label, svd in enumerate(class_val_svd):
                 plt.plot(np.cumsum(svd)/svd.sum(), label=f'Class {label}')
            plt.xlabel("Sorted singular value index")
            plt.ylabel('Cumulative Explained Variance')
            plt.title('Validation')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'class_specific_collapse.png'), dpi=300)
        plt.cla()

        with open(os.path.join(save_path, 'class_specific_auc.csv'), 'a') as f:
            if val_activations is not None:
                f.write(f'class,tr_svd,val_auc,test_auc\n')
                for label, (tr_svd, val_svd, test_svd) in enumerate(zip(class_tr_svd, class_val_svd, class_test_svd)):
                    f.write(f'{label},{auc(tr_svd):.4f},{auc(val_svd):.4f},{auc(test_svd):.4f}\n')
            else:
                f.write(f'class,tr_svd,test_auc\n')
                for label, test_svd in enumerate(class_tr_svd, class_test_svd):
                    f.write(f'{label},{auc(tr_svd):.4f},{auc(test_svd):.4f}\n')




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