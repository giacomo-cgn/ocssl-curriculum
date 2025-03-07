import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

def prototype_drift(tr_activations: torch.Tensor,
                     tr_labels: torch.Tensor,
                     test_activations: torch.Tensor,
                     test_labels: torch.Tensor,
                     val_activations: torch.Tensor = None,
                     val_labels: torch.Tensor = None,
                     save_path: str = None,
                     pretr_exp_idx: int = 0,
                     previous_classes: list = None,
                     current_classes: list = None,
              ):
    
    tr_activations = tr_activations.cpu()
    tr_labels = tr_labels.cpu()
    test_activations = test_activations.cpu()
    test_labels = test_labels.cpu()
    if val_activations is not None:
        val_activations = val_activations.cpu()
        val_labels = val_labels.cpu()

    current_save_path = os.path.join(save_path, f'pretr_exp_{pretr_exp_idx}')
    if not os.path.exists(current_save_path):
        os.makedirs(current_save_path)

    # Calculate current labels prototypes
    tr_prototypes = torch.zeros(tr_labels.max().item() + 1, tr_activations.shape[1])
    for label in tr_labels.unique():
        tr_prototypes[label] = tr_activations[tr_labels == label].mean(dim=0)
    torch.save(tr_prototypes, os.path.join(current_save_path, 'tr_prototypes.pt'))
    
    test_prototypes = torch.zeros(test_labels.max().item() + 1, test_activations.shape[1])
    for label in test_labels.unique():
        test_prototypes[label] = test_activations[test_labels == label].mean(dim=0)
    torch.save(test_prototypes, os.path.join(current_save_path, 'test_prototypes.pt'))

    if val_activations is not None:
        val_prototypes = torch.zeros(val_labels.max().item() + 1, val_activations.shape[1])
        for label in val_labels.unique():
            val_prototypes[label] = val_activations[val_labels == label].mean(dim=0)
        torch.save(val_prototypes, os.path.join(current_save_path, 'val_prototypes.pt'))

    # Recover previous evaluation prototypes
    if pretr_exp_idx > 0:
        prev_save_path = os.path.join(save_path, f'pretr_exp_{pretr_exp_idx - 1}')
        prev_tr_prototypes = torch.load(os.path.join(prev_save_path, 'tr_prototypes.pt'))
        if val_activations is not None:
            prev_val_prototypes = torch.load(os.path.join(prev_save_path, 'val_prototypes.pt'))
        prev_test_prototypes = torch.load(os.path.join(prev_save_path, 'test_prototypes.pt'))

        # Compute prototype drift    
        tr_prototype_drift = torch.norm(tr_prototypes - prev_tr_prototypes, dim=1)
        test_prototype_drift = torch.norm(test_prototypes - prev_test_prototypes, dim=1)
        if val_activations is not None:
            val_prototype_drift = torch.norm(val_prototypes - prev_val_prototypes, dim=1)

        # Save drift in .csv format
        drift_df = pd.DataFrame({'class': tr_labels.unique().numpy(),
                                  'tr_prototype_drift': tr_prototype_drift.numpy(),
                                  'test_prototype_drift': test_prototype_drift.numpy()})
        if val_activations is not None:
            drift_df['val_prototype_drift'] = val_prototype_drift.numpy()
        drift_df.to_csv(os.path.join(current_save_path, 'prototype_drift.csv'), index=False)

        # Save drift metrics
        with open(os.path.join(current_save_path, 'drift_metrics.csv'), 'a') as f:
            if val_activations is not None:
                f.write('metric,tr,val,test\n')
                f.write(f'tot_drift,{tr_prototype_drift.mean().item():.2f},{val_prototype_drift.mean().item():.2f},{test_prototype_drift.mean().item():.2f}\n')
                f.write(f'current_classes_drift,{tr_prototype_drift[current_classes].mean().item():.2f},{val_prototype_drift[current_classes].mean().item():.2f},{test_prototype_drift[current_classes].mean().item():.2f}\n')

            else:
                f.write('metric,tr,test\n')
                f.write(f'tot_drift,{tr_prototype_drift.mean().item():.2f},{test_prototype_drift.mean().item():.2f}\n')
                f.write(f'current_classes_drift,{tr_prototype_drift[current_classes].mean().item():.2f},{test_prototype_drift[current_classes].mean().item():.2f}\n')


            if previous_classes is not None:
                new_classes = list(set(current_classes) - set(previous_classes))
                old_classes = list(set(previous_classes) - set(current_classes))
                common_classes = list(set(previous_classes) & set(current_classes))
                if val_activations is not None:
                    if len(new_classes) > 0:
                        f.write(f'new_classes_drift,{tr_prototype_drift[new_classes].mean().item():.2f},{val_prototype_drift[new_classes].mean().item():.2f},{test_prototype_drift[new_classes].mean().item():.2f}\n')
                    if len(old_classes) > 0:
                        f.write(f'old_classes_drift,{tr_prototype_drift[old_classes].mean().item():.2f},{val_prototype_drift[old_classes].mean().item():.2f},{test_prototype_drift[old_classes].mean().item():.2f}\n')
                    if len(common_classes) > 0:
                        f.write(f'common_classes_drift,{tr_prototype_drift[common_classes].mean().item():.2f},{val_prototype_drift[common_classes].mean().item():.2f},{test_prototype_drift[common_classes].mean().item():.2f}\n')
                    f.write(f'previous_classes_drift,{tr_prototype_drift[previous_classes].mean().item():.2f},{val_prototype_drift[previous_classes].mean().item():.2f},{test_prototype_drift[previous_classes].mean().item():.2f}\n')
                else:
                    if len(new_classes) > 0:
                        f.write(f'new_classes_drift,{tr_prototype_drift[new_classes].mean().item():.2f},{test_prototype_drift[new_classes].mean().item():.2f}\n')
                    if len(old_classes) > 0:
                        f.write(f'old_classes_drift,{tr_prototype_drift[old_classes].mean().item():.2f},{test_prototype_drift[old_classes].mean().item():.2f}\n')
                    if len(common_classes) > 0:
                        f.write(f'common_classes_drift,{tr_prototype_drift[common_classes].mean().item():.2f},{test_prototype_drift[common_classes].mean().item():.2f}\n')
                    f.write(f'previous_classes_drift,{tr_prototype_drift[previous_classes].mean().item():.2f},{test_prototype_drift[previous_classes].mean().item():.2f}\n')


def drift_plot(drift_pth: str):
    # # List all subfolders named 'pretr_exp_{i}' in the given path and order by i
    # subfolders = sorted([f for f in os.listdir(drift_pth) if f.startswith('pretr_exp_')], key=lambda x: int(x.split('_')[-1]))

    # exp_drift_df_list = []
    # for subfolder in subfolders:

    #     drift_metrics_df = pd.read_csv(os.path.join(drift_pth, subfolder, 'drift_metrics.csv'))
        
    from collections import defaultdict
    
    # Initialize data structure to hold all metrics
    data = defaultdict(lambda: defaultdict(dict))  # metric -> split -> {exp_idx: value}

    experience_indices = []

    # Collect data from all experience folders
    i = 1  # Starting from 1 as per your description
    while True:
        exp_dir = os.path.join(drift_pth, f"pretr_exp_{i}")
        csv_path = os.path.join(exp_dir, "drift_metrics.csv")
        
        if not os.path.exists(csv_path):
            break
        
        try:
            df = pd.read_csv(csv_path, skipinitialspace=True)
            splits = [col for col in df.columns if col != 'metric']
            
            for _, row in df.iterrows():
                metric = row['metric'].strip()
                for split in splits:
                    value = row[split]
                    data[metric][split][i] = value
                    
            experience_indices.append(i)
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
        
        i += 1

    # Generate plots
    metrics = sorted(data.keys())
    if not metrics:
        print("No metrics found in any files")
        exit()

    ncols = 3
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5*nrows))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        splits_present = [split for split in ['tr', 'val', 'test'] 
                        if split in data[metric] and data[metric][split]]
        
        for split in splits_present:
            split_data = data[metric][split]
            sorted_exps = sorted(split_data.keys())
            values = [split_data[exp] for exp in sorted_exps]
            ax.plot(sorted_exps, values, 'o-', label=split)  # Swapped x and y
        
        ax.set_title(metric)
        ax.set_xlabel("Experience Index")  # Updated label
        ax.set_ylabel("Metric Value")       # Updated label
        ax.grid(True)
        ax.legend()

    # Hide unused subplots
    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(drift_pth, 'drift_plot.png'))
    plt.cla()





            

            
               
            
        


    
    

