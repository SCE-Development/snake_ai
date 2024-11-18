import torch.nn as nn
import torch


class PPOModel(nn.Module):
    def __init__(self): ...

    def get_action(self, state: torch.Tensor) -> int:
        """
        Returns an integer corresponding to the action to take
        """

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns a tensor containing a single item corresponding
        to the value function evaluated for state
        """
