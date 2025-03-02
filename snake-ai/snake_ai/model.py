"""
File: model.py

This file contains the code for a shared-weights actor-critic model.
"""

from typing import Tuple
import torch.nn as nn
import torch
from abc import abstractmethod
from copy import deepcopy


class ActorCritic(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calls the model to predict the action probabilities and values B states

        Arguments:
            states (torch.Tensor): the states to predict action probabilities
                for. Shape should be (B, *state_dims)

        Returns:
            items (Tuple[torch.Tensor, torch.Tensor]): a tuple of (log_probabilities, values), where
                log_probabilities is a tensor of the shape (B, action_space) and values is a tensor
                of the shape (B,)
        """

    def predict(
        self, state: torch.Tensor, deterministic: bool = True
    ) -> Tuple[int, float, float]:
        """
        Convenience method to predict the (discrete) action
        to take, its log probability, and its value given a nonbatched state.

        Arguments:
            state (torch.Tensor): the current state, nonbatched. Shape
                should be (*state_dims)
        Returns:
            items (Tuple[int, float, float]): the discrete action to take, its log_probability, and its predicted value
        """
        actions, log_probs, vals = self.predict_batched(
            state.unsqueeze(0), deterministic
        )
        return (
            actions.squeeze().item(),
            log_probs.squeeze().item(),
            vals.squeeze().item(),
        )

    def predict_batched(
        self, state: torch.Tensor, deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method to predict the (discrete) action to take, its log probability, and its value
        given the batched current states

        Arguments:
            state (torch.Tensor): the current state, batched. Shape
                should be (B, *state_dims)
        Returns:
            items (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): the discrete actions to take (B,),
                their log_probability (B,), and their respective predicted values (B,)
        """
        with torch.no_grad():
            log_probs, vals = self(state)
        if deterministic:
            actions = log_probs.argmax(dim=-1)
        else:
            actions = torch.exp(log_probs).multinomial(1).squeeze(dim=-1)

        return (
            actions,
            log_probs[torch.arange(log_probs.shape[0]), actions],
            vals.squeeze(dim=-1),
        )


class CNNActorCritic(ActorCritic):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_actions: int,
        kernel_size: int,
        conv_layers: int,
        hidden_dim: int,
        dropout_rate: float = 0.1,
    ):
        # convolutional network
        super(CNNActorCritic, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)

        h, w, c = input_shape
        self.dropout = nn.Dropout(dropout_rate)
        convs = []
        for _ in range(conv_layers):
            conv = nn.Conv2d(
                c,
                c * 2,
                kernel_size=kernel_size,
                stride=2,
                padding=0,
                dilation=1,
            )
            h = int((h - (kernel_size - 1) - 1) / 2 + 1)
            w = int((w - (kernel_size - 1) - 1) / 2 + 1)
            c *= 2
            convs.append(conv)
            convs.append(nn.SELU())

        self.prob_module = nn.Sequential(
            *convs,
            nn.Dropout(dropout_rate),
            nn.Flatten(start_dim=1),
            nn.Linear(h * w * c, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_actions),
        )
        self.val_module = nn.Sequential(
            *deepcopy(convs),
            nn.Dropout(dropout_rate),
            nn.Flatten(start_dim=1),
            nn.Linear(h * w * c, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

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

        # B, H, W, C -> B, C, H, W
        x = states.float().permute(0, 3, 1, 2).contiguous()

        # calculate probability logits and value
        log_probs = self.log_softmax(self.prob_module(x))
        vals = self.val_module(x)
        return log_probs, vals


class MLPActorCritic(ActorCritic):
    def __init__(
        self,
        num_actions: int,
        state_dim: int,
        hidden_dim: int,
    ):
        # simple mlp network
        super(MLPActorCritic, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=1)

        self.shared = nn.Sequential()

        self.prob_dense = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )
        self.val_dense = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calls the model to predict the action probabilities and values B states

        Arguments:
            states (torch.Tensor): the states to predict action probabilities
                for. Shape should be (B, *state_dims)

        Returns:
            items (Tuple[torch.Tensor, torch.Tensor]): a tuple of (log_probabilities, values), where
                log_probabilities is a tensor of the shape (B, action_space) and values is a tensor
                of the shape (B,)
        """

        x = self.shared(states)
        # calculate probability logits and value
        log_probs = self.log_softmax(self.prob_dense(x))
        vals = self.val_dense(x)
        return log_probs, vals
