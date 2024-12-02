"""
File: model.py

This file contains the code for a shared-weights actor-critic model.
"""

from typing import Tuple
import torch.nn as nn
import torch


class ActorCritic(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ActorCritic, self).__init__(*args, **kwargs)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calls the model to predict the action probabilities and values B states

        Arguments:
            states (torch.Tensor): the states to predict action probabilities
                for. Shape should be (B, *state_dims)

        Returns:
            items (Tuple[torch.Tensor, torch.Tensor]): a tuple of (probabilities, values), where
                probabilities is a tensor of the shape (B, action_space) and values is a tensor
                of the shape (B,)
        """
        # TODO - implement me

    def predict(self, state: torch.Tensor) -> Tuple[int, float]:
        """
        Convenience method to predict the (discrete) action
        to take given a nonbatched state, its probability, and its value.

        Arguments:
            state (torch.Tensor): the current state, nonbatched. Shape
                should be (*state_dims)
        Returns:
            items (Tuple[int, float, float]): the discrete action to take, its probability, and its predicted value
        """
        with torch.no_grad():
            probs, vals = self(state.unsqueeze(0))
        probs = probs.squeeze()
        argmax = probs.argmax()
        return argmax.item(), probs[argmax].item(), vals.squeeze().item()
