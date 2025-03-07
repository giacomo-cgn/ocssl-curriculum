import torch

class AbstractProbe():
    def __init__(self) -> None:
        pass

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
        raise NotImplementedError
    
    def get_name(self) -> str:
        raise NotImplementedError