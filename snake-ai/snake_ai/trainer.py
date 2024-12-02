"""
File: trainer.py

This file contains the code for training an actor-critic
model using the PPO algorithm.
"""

import torch
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
from gymnasium import Env
from typing import List
from .model import ActorCritic
from dataclasses import dataclass, field
from bisect import bisect_right
from tqdm import tqdm


@dataclass
class EpisodeRecord:
    states: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    actions: torch.Tensor
    probs: torch.Tensor
    advantages: torch.Tensor = field(init=False)
    T: int = field(init=False)

    def __post_init__(self):
        self.T = self.states.shape[0]
        self.advantages = None

    def calculate_advantages(self, gamma: float, lam: float):
        # deltas[i] = how much better state i+1 is compared to state i
        deltas = []
        for t in range(self.T - 1):
            deltas.append(self.rewards[t] + gamma * self.values[t + 1] - self.values[t])
        # instead of doing the N^2 version of calculating advantages,
        # go from the back instead
        advantages = torch.zeros((len(deltas),))
        # set the last item, -2 since 0-indexed
        advantages[self.T - 2] = deltas[self.T - 2]
        for t in range(self.T - 3, -1, -1):
            advantages[t] = deltas[t] + (lam * gamma) * advantages[t + 1]
        self.advantages = advantages


class EpisodeDataset(Dataset):
    def __init__(self, records: List[EpisodeRecord]):
        super().__init__()
        self.records = records

        # figure out what index every episode should start at
        self.starts = []
        cur = 0
        for record in records:
            self.starts.append(cur)
            # there's actually only T-1 observations (since only T-1 advantages)
            cur += record.T - 1
        self.total = cur  # total amount of observations

    def __getitem__(self, index):
        record_idx = bisect_right(self.starts, index) - 1
        episode_idx = index - self.starts[record_idx]
        item = self.records[record_idx]
        return {
            "states": item.states[episode_idx],
            "values": item.values[episode_idx],
            "actions": item.actions[episode_idx],
            "probs": item.probs[episode_idx],
            "rewards": item.rewards[episode_idx],
            "advantages": item.advantages[episode_idx],
        }

    def __len__(self):
        return self.total


def collect_samples(
    samples: int, model: ActorCritic, env: Env, device: str
) -> List[EpisodeRecord]:
    """
    Collects `samples` samples from the environment, resetting it at first
    Assumes that model.get_value and model.get_action have gradients

    Arguments:
        samples (int): the number of samples to collect
        model (ActorCritic): the actor-critic
        env (Env): the environment to use
        device (str): the device to use

    Returns:
        EpisodeRecord: the states, values, and rewards
    """
    cur_states = []
    cur_values = []
    cur_rewards = []
    cur_actions = []
    cur_probs = []
    records = []

    obs, _ = env.reset()
    num_samples = 0
    cur_length = 0
    while num_samples < samples:
        # take current and prepare to step
        cur_states.append(obs)
        action, prob, value = model.predict(
            torch.tensor(obs).to(device), deterministic=False
        )
        cur_values.append(value)
        cur_actions.append(action)
        cur_probs.append(prob)

        # step in the environment
        obs, reward, terminated, truncated, _ = env.step(action)
        cur_rewards.append(reward)

        if terminated or truncated:
            if cur_length > 2:
                # new episode
                records.append(
                    EpisodeRecord(
                        states=torch.tensor(np.array(cur_states)),
                        values=torch.tensor(cur_values),
                        rewards=torch.tensor(cur_rewards),
                        actions=torch.tensor(cur_actions),
                        probs=torch.tensor(cur_probs),
                    )
                )
            cur_rewards = []
            cur_states = []
            cur_values = []
            cur_length = 0
            # reset the environment
            obs, _ = env.reset()
        num_samples += 1
        cur_length += 1

    # in case we have extra remaining that we didn't include in the list yet
    if cur_length > 2:
        # new episode
        records.append(
            EpisodeRecord(
                states=torch.tensor(np.array(cur_states)),
                values=torch.tensor(cur_values),
                rewards=torch.tensor(cur_rewards),
                actions=torch.tensor(cur_actions),
                probs=torch.tensor(cur_probs),
            )
        )
    return records


def train(
    model: ActorCritic,
    optimizer: Optimizer,
    env: Env,
    iterations: int,
    samples: int,
    batch_size: int,
    epochs: int,
    gamma: float,
    lam: float,
    eps: float,
    vcf: float = 1.0,
    ecf: float = 1.0,
    device: str = "cpu",
    num_workers: int = 4,
):
    # see page 5 of https://arxiv.org/pdf/1707.06347
    """
    Train an actor-critic model in the given environment using the PPO algorithm.

    Arguments:
        model (ActorCritic): the actor-critic model to train
        optimizer (Optimizer): the optimizer to use
        env (Env): the environment to train in
        iterations (int): the number of iterations to train
        samples (int): the number of samples to collect per iteration
        batch_size (int): the batch size for training
        epochs (int): the number of epochs to train per iteration
        gamma (float): the discount factor
        lam (float): the lambda parameter for GAE
        eps (float): the epsilon parameter for clipping
        vcf (float, optional): the value function coefficient. Defaults to 1.0.
        ecf (float, optional): the entropy coefficient. Defaults to 1.0.
        device (str, optional): the device to use (cpu or cuda). Defaults to "cpu".
    """
    model.to(device)
    for iteration in range(iterations):
        print(f"iteration {iteration+1}")
        model.eval()
        episodes = collect_samples(samples, model, env, device)

        # calculate advantages
        for episode in episodes:
            episode.calculate_advantages(gamma, lam)

        # fit actor and critic
        model.train()
        ds = EpisodeDataset(episodes)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        prog = tqdm(total=len(loader) * epochs)
        total_loss = 0
        total_lclip = 0
        total_lvf = 0
        for _ in range(epochs):
            for batch in loader:
                # clear old gradients
                optimizer.zero_grad()

                # get batch data
                states, actions, probs, rewards, advantages = (
                    batch["states"].to(device),
                    batch["actions"].to(device),
                    batch["probs"].to(device),
                    batch["rewards"].to(device),
                    batch["advantages"].to(device),
                )

                # predict
                pred_probs, pred_vals = model(states)

                # calculate lclip
                cur_action_probs = pred_probs[
                    torch.arange(pred_probs.shape[0]), actions
                ]
                ratio = cur_action_probs / probs.where(probs == 0, 1e-8)
                # paper says we want to maximize this, so therefore just negate it
                lclip = -torch.minimum(
                    ratio * advantages, torch.clip(ratio, 1 - eps, 1 + eps) * advantages
                ).mean()

                # calculate lvf
                lvf = torch.mean((pred_vals - rewards) ** 2)

                # calculate entropy bonus
                # encourage low probabilities for the current actions
                # (log of probability is a negative number, smaller probability -> smaller log)
                lentropy = torch.log(cur_action_probs).mean()

                # backpropagate
                loss = lclip + vcf * lvf + ecf * lentropy
                loss.backward()
                optimizer.step()

                # logging
                loss = loss.detach().item()
                lclip = lclip.detach().item()
                lvf = lvf.detach().item()
                lentropy = lentropy.detach().item()

                prog.set_postfix(
                    {
                        "loss": loss,
                        "lclip": lclip,
                        "lvf": lvf,
                        "lentropy": lentropy,
                    }
                )
                prog.update()
                prog.display()
                total_loss += loss
                total_lclip += lclip
                total_lvf += lvf
                if np.isnan(loss):
                    print(pred_probs)
                    print(pred_vals)
        prog.close()
        print(
            f"Total loss {total_loss:.4f}, lclip {total_lclip:.4f}, lvf {total_lvf:.4f}"
        )

    model.eval()
