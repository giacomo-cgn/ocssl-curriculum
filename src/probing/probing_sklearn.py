import torch
import torch.nn as nn

from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from .abstract_probe import AbstractProbe

class ProbingSklearn(AbstractProbe):
    def __init__(self,
                 probe_type: str = 'rr',
                 knn_k: int = 50,
                 device: str = 'cpu',
                 mb_size: int = 512,
                 seed: int = 42,
                 config_save_pth: str = None
                 ):
        
        self.probe_type = probe_type
        self.knn_k = knn_k
        self.device = device
        self.mb_size = mb_size
        self.seed = seed

        self.criterion = nn.CrossEntropyLoss()

        if config_save_pth is not None:
            # Save model configuration
            with open(config_save_pth + '/config.txt', 'a') as f:
                # Write strategy hyperparameters
                f.write('\n')
                f.write('---- PROBE CONFIG ----\n')
                f.write(f'Probing type: {probe_type}\n')
                if probe_type == 'knn':
                    f.write(f'KNN k: {knn_k}\n')
                f.write(f'Eval MB size: {mb_size}\n')

    def get_name(self):
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
        
        tr_activations = tr_activations.cpu().numpy()
        tr_labels = tr_labels.cpu().numpy()
        if val_activations is not None:
            val_activations = val_activations.cpu().numpy()
            val_labels = val_labels.cpu().numpy()
        test_activations = test_activations.cpu().numpy()
        test_labels = test_labels.cpu().numpy()


        scaler = StandardScaler()
        
        # Use a simple classifier on activations
        if self.probe_type == 'rr':
            clf = RidgeClassifier()
        elif self.probe_type == 'knn':
            clf = KNeighborsClassifier(n_neighbors=self.knn_k)

        tr_activations = scaler.fit_transform(tr_activations)
        clf = clf.fit(tr_activations, tr_labels)
        
        if val_activations is not None:
            # Predict validation
            val_activations = scaler.transform(val_activations)
            val_preds = clf.predict(val_activations)
            # Calculate validation accuracy and loss
            val_acc = accuracy_score(val_labels, val_preds)

        # Predict test
        test_activations = scaler.transform(test_activations)
        test_preds = clf.predict(test_activations)
        # Calculate test accuracy
        test_acc = accuracy_score(test_labels, test_preds)

        if save_file is not None:
            with open(save_file, 'a') as f:
                if val_activations is not None:
                    if exp_idx is not None:
                        f.write(f'{exp_idx},_,{test_acc:.4f}\n')
                    else:
                        f.write(f'_,{test_acc:.4f}\n')
                else:        
                    if exp_idx is not None:
                        f.write(f'{exp_idx},{val_acc:.4f},{test_acc:.4f}\n')
                    else:
                        f.write(f'{val_acc:.4f},{test_acc:.4f}\n')
