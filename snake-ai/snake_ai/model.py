"""
File: model.py

This file contains the code for a shared-weights actor-critic model.
"""

from typing import Tuple
import torch.nn as nn
import torch
from abc import abstractmethod


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
            items (Tuple[torch.Tensor, torch.Tensor]): a tuple of (probabilities, values), where
                probabilities is a tensor of the shape (B, action_space) and values is a tensor
                of the shape (B,)
        """

    def predict(
        self, state: torch.Tensor, deterministic: bool = True
    ) -> Tuple[int, float, float]:
        """
        Convenience method to predict the (discrete) action
        to take, its probability, and its value given a nonbatched state.

        Arguments:
            state (torch.Tensor): the current state, nonbatched. Shape
                should be (*state_dims)
        Returns:
            items (Tuple[int, float, float]): the discrete action to take, its probability, and its predicted value
        """
        with torch.no_grad():
            probs, vals = self(state.unsqueeze(0))
        probs: torch.Tensor = probs.squeeze()
        if deterministic:
            idx = probs.argmax()
        else:
            idx = probs.multinomial(1)
        return idx.item(), probs[idx].item(), vals.squeeze().item()

    def predict_batched(
        self, state: torch.Tensor, deterministic: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convenience method to predict the (discrete) action to take, its probability, and its value
        given the batched current states

        Arguments:
            state (torch.Tensor): the current state, batched. Shape
                should be (B, *state_dims)
        Returns:
            items (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): the discrete actions to take (B,),
                their probability (B,), and their respective predicted values (B,)
        """
        with torch.no_grad():
            probs, vals = self(state)
        if deterministic:
            actions = probs.argmax(dim=-1)
        else:
            actions = probs.multinomial(1).squeeze(dim=-1)

        return (
            actions,
            probs[torch.arange(probs.shape[0]), actions],
            vals.squeeze(dim=-1),
        )


class CNNActorCritic(ActorCritic):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_actions: int,
        emb_dim: int,
        kernel_size: int,
        hidden_dim: int,
        dropout_rate: float = 0.1,
    ):
        # convolutional network
        super(CNNActorCritic, self).__init__()

        self.gelu = nn.GELU()
        self.selu = nn.SELU()
        self.softmax = nn.Softmax(dim=1)

        h, w, c = input_shape
        self.dropout = nn.Dropout(dropout_rate)
        self.preconv = nn.Linear(input_shape[-1], emb_dim)
        c = emb_dim
        self.convs = []
        while h * w * c > hidden_dim * 2:
            conv = nn.Conv2d(
                c, c // 2, kernel_size=kernel_size, stride=2, padding=0, dilation=1
            )
            h = int((h - (kernel_size - 1) - 1) / 2 + 1)
            w = int((w - (kernel_size - 1) - 1) / 2 + 1)
            c //= 2
            self.convs.append(conv)
        self.convs = nn.ModuleList(self.convs)
        self.prob_dense = nn.Sequential(
            nn.Linear(h * w * c, hidden_dim),
            self.gelu,
            nn.Linear(hidden_dim, hidden_dim),
            self.gelu,
            nn.Linear(hidden_dim, num_actions),
        )
        self.val_dense = nn.Sequential(
            nn.Linear(h * w * c, hidden_dim),
            self.gelu,
            nn.Linear(hidden_dim, hidden_dim),
            self.gelu,
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

        x: torch.Tensor = self.gelu(self.preconv(states.float()))
        # B, H, W, C -> B, C, H, W
        x = x.permute(0, 3, 1, 2).contiguous()

        # pass through shared convs
        for conv in self.convs:
            x = self.dropout(self.selu(conv(x)))
        x = x.flatten(start_dim=1)

        # calculate probability logits and value
        probs = self.softmax(self.prob_dense(x))
        vals = self.val_dense(x)
        return probs, vals


class MLPActorCritic(ActorCritic):
    def __init__(
        self,
        num_actions: int,
        state_dim: int,
        hidden_dim: int,
    ):
        # simple mlp network
        super(MLPActorCritic, self).__init__()

        self.softmax = nn.Softmax(dim=1)

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.prob_dense = nn.Sequential(
            nn.Linear(hidden_dim, num_actions),
        )
        self.val_dense = nn.Sequential(
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

        x = self.shared(states)
        # calculate probability logits and value
        probs = self.softmax(self.prob_dense(x))
        vals = self.val_dense(x)
        return probs, vals
