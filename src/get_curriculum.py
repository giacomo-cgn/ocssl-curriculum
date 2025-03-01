
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset, Dataset, random_split
from sklearn.model_selection import train_test_split

from .get_datasets import get_benchmark, get_iid_dataset

class CurriculumTask:
    def __init__(self, dataset, task_name, tr_steps):
        self.task_name = task_name
        self.tr_steps = int(tr_steps)
        self.dataset = dataset



def get_curriculum(curriculum_order_list,
                   dataset_name,
                   total_steps,
                   kwargs   
                ):
    
    benchmark, image_size = get_benchmark(dataset_name, kwargs["dataset_root"], num_exps=2, seed=kwargs["dataset_seed"], val_ratio=kwargs["val_ratio"])
    iid_dataset = get_iid_dataset(benchmark)
    
    curriculum = []
    # Init the Subset splitter
    subset_splitter = SubsetSplitterWrapper(seed=kwargs["dataset_seed"])

    tot_ratio = 0.0
    for curriculum_part in curriculum_order_list:
        tot_ratio += curriculum_part['steps_ratio']
    assert tot_ratio == 1.0, f'The sum of the steps ratios in the curriculum order list is {tot_ratio}, but it should be 1.0'
        

    for i, curriculum_part in enumerate(curriculum_order_list):
        # Full IID task
        if curriculum_part['type'] == 'iid':
            steps = curriculum_part['steps_ratio'] * total_steps
            curriculum.append(CurriculumTask(iid_dataset, 'iid', steps))
        
        # Subset IID task
        elif curriculum_part['type'] == 'subset':
            subset_ratio = curriculum_part['subset_ratio']
            subset_len = int(subset_ratio*len(iid_dataset))
            subset, _ = subset_splitter.subset(iid_dataset, subset_len, len(iid_dataset) - subset_len)
            steps = curriculum_part['steps_ratio'] * total_steps
            curriculum.append(CurriculumTask(subset, f'subset: {subset_ratio}', steps))
        
        # Subset cumulative task, where the subset ratio increases linearly
        elif curriculum_part['type'] == 'subset_cumulative':
            assert curriculum_part['start_subset_ratio'] > 0.0, 'The start subset ratio should be greater than 0.0'
            steps = curriculum_part['steps_ratio'] * total_steps / curriculum_part['num_tasks']
            ratio_increment = (curriculum_part['end_subset_ratio'] - curriculum_part['start_subset_ratio']) / (curriculum_part['num_tasks']-1)
            num_tasks = curriculum_part['num_tasks'] - 1

            start_subset_len = int(curriculum_part['start_subset_ratio']*len(iid_dataset))
            current_subset, _ = subset_splitter.subset(iid_dataset, start_subset_len, len(iid_dataset) - start_subset_len)
            curriculum.append(CurriculumTask(current_subset, f'cumulative subset: {curriculum_part["start_subset_ratio"]}', steps))

            for subset_ratio in enumerate(np.linspace(curriculum_part['start_subset_ratio']+ratio_increment, curriculum_part['end_subset_ratio'], num_tasks)):
                additional_len = int(ratio_increment*len(iid_dataset))
                remaining_indices = list(set(range(len(iid_dataset))) - set(current_subset.indices))
                remaining_dataset = Subset(iid_dataset, remaining_indices)
                new_samples, _ = subset_splitter.subset(remaining_dataset, additional_len, len(remaining_dataset) - additional_len)
                current_subset = Subset(iid_dataset, current_subset.indices + new_samples.indices)
                curriculum.append(CurriculumTask(current_subset, f'cumulative subset {j}: {subset_ratio}', steps))  

        # Standard Continual Class incremental
        elif curriculum_part['type'] == 'class_incremental':
            cl_benchmark, _ = get_benchmark(dataset_name, kwargs["dataset_root"], num_exps=curriculum_part['num_tasks'], seed=kwargs["dataset_seed"], val_ratio=kwargs["val_ratio"])
            for j, exp_dataset in enumerate(cl_benchmark.train_stream):
                steps = curriculum_part['steps_ratio'] * total_steps / curriculum_part['num_tasks']
                curriculum.append(CurriculumTask(exp_dataset, f'class_incremental: {j}', steps))

        # Cumulative Continual Class incremental, each task is the sum of the previous ones
        elif curriculum_part['type'] == 'class_cumulative':
            steps = curriculum_part['steps_ratio'] * total_steps / curriculum_part['num_tasks']
            cl_benchmark, _ = get_benchmark(dataset_name, kwargs["dataset_root"], num_exps=curriculum_part['num_tasks'], seed=kwargs["dataset_seed"], val_ratio=kwargs["val_ratio"])
            # sum each dataset task to the previous one
            cum_dataset = cl_benchmark.train_stream[0]
            curriculum.append(CurriculumTask(cum_dataset, f'class_cumulative: 0', steps))
            for j, exp_dataset in enumerate(cl_benchmark.train_stream[1:]):
                cum_dataset = ConcatDataset([cum_dataset, exp_dataset])
                curriculum.append(CurriculumTask(cum_dataset, f'class_cumulative: {j+1}', steps))

        else:
            raise ValueError(f'Unknown curriculum type: {curriculum_part["type"]}')

    for i, curriculum_task in enumerate(curriculum):
        print(f'Task {i} {curriculum_task.task_name}, steps {curriculum_task.tr_steps}, len {len(curriculum_task.dataset)}')

    return curriculum, image_size


class SubsetSplitterWrapper:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def subset(self, dataset: Dataset, len1: int, len2: int, subset_type: str = 'random'):
        if subset_type == 'random':
            subset_1, subset2 = random_split(dataset, [len1, len2], generator=torch.Generator().manual_seed(self.seed))
        elif subset_type == 'class_balanced':
            subset_1, subset2 = class_balanced_split(dataset, len1, len2, self.seed)
        else:
            raise ValueError("Invalid splitter type. Choose 'random' or 'class_balanced'.")
        return subset_1, subset2


def class_balanced_split(dataset: Dataset, len1: int, len2: int, seed: int = 42):
    if len1 + len2 > len(dataset):
        raise ValueError("The sum of len1 and len2 exceeds the total number of samples in the dataset.")

    indices = list(range(len(dataset)))
    class_labels = [dataset[idx][1] for idx in indices]

    if len1 == 0 or len2 == 0:
        return Subset(dataset, indices[:len1]), Subset(dataset, indices[len1:])

    subset_indices_1, subset_indices_2 = train_test_split(indices, test_size=len2 / (len1 + len2), stratify=class_labels, random_state=seed)

    return Subset(dataset, subset_indices_1), Subset(dataset, subset_indices_2)