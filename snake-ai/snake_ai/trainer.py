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
from typing import List, Tuple
from .model import ActorCritic
from dataclasses import dataclass, field
from bisect import bisect_right
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter


@dataclass
class EpisodeRecord:
    states: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    actions: torch.Tensor
    probs: torch.Tensor
    advantages: torch.Tensor = field(init=False)
    target_values: torch.Tensor = field(init=False)
    T: int = field(init=False)

    def __post_init__(self):
        self.T = self.states.shape[0]
        self.advantages = None
        self.target_values = None

    def calculate(self, gamma: float, lam: float):
        # deltas[i] = how much better state i+1 is compared to state i
        deltas = []
        for t in range(self.T - 1):
            deltas.append(self.rewards[t] + gamma * self.values[t + 1] - self.values[t])
        deltas.append(0)

        # instead of doing the N^2 version of calculating advantages,
        # go from the back instead
        advantages = torch.zeros((len(deltas),))
        # set the last item
        advantages[-1] = deltas[-1]
        for t in range(self.T - 2, -1, -1):
            advantages[t] = deltas[t] + (lam * gamma) * advantages[t + 1]
        self.advantages = advantages

        # also calculate target values
        # target value should just be the discounted future reward
        self.target_values = self.advantages + self.values


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
            "target_values": item.target_values[episode_idx],
            "actions": item.actions[episode_idx],
            "probs": item.probs[episode_idx],
            "rewards": item.rewards[episode_idx],
            "advantages": item.advantages[episode_idx],
        }

    def __len__(self):
        return self.total


def collect_samples(
    samples: int, t: int, model: ActorCritic, env: Env, device: str
) -> Tuple[List[EpisodeRecord], float, float]:
    """
    Collects `samples` samples from the environment, resetting it at first
    Assumes that model.get_value and model.get_action have gradients

    Arguments:
        samples (int): the number of samples to collect
        t (int): the maximum number of samples to collect before resetting the env
        model (ActorCritic): the actor-critic
        env (Env): the environment to use
        device (str): the device to use

    Returns:
        ret (Tuple[List[EpisodeRecord], float, float): a tuple containing
            the list of episode records, the mean reward, and the mean
            episode length
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
    cur_reward = 0
    total_reward = 0
    total_length = 0
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
        cur_reward += reward

        if terminated or truncated or cur_length >= t:
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
                total_reward += cur_reward
                total_length += cur_length

            cur_rewards = []
            cur_states = []
            cur_values = []
            cur_length = 0
            cur_reward = 0
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
        total_reward += cur_reward
        total_length += cur_length
    return records, total_reward / len(records), total_length / len(records)


def train(
    model: ActorCritic,
    optimizer: Optimizer,
    env: Env,
    iterations: int,
    t: int,
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
    clip_norm: bool = True,
    clip_norm_val: float = 2.0,
    normalize_advantages: bool = False,
):
    # see page 5 of https://arxiv.org/pdf/1707.06347
    """
    Train an actor-critic model in the given environment using the PPO algorithm.

    Arguments:
        model (ActorCritic): the actor-critic model to train
        optimizer (Optimizer): the optimizer to use
        env (Env): the environment to train in
        iterations (int): the number of iterations to train
        t (int): the number of samples to take from each environment
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
    i = 0
    writer = SummaryWriter()
    writer.add_scalar("params/gamma", gamma, 0)
    writer.add_scalar("params/lambda", lam, 0)
    writer.add_scalar("params/vcf", vcf, 0)
    writer.add_scalar("params/ecf", ecf, 0)
    writer.add_scalar("params/batch_size", batch_size, 0)
    writer.add_scalar("params/epochs", epochs, 0)

    for iteration in range(iterations):
        print(f"iteration {iteration+1}")
        model.eval()
        episodes, mean_reward, mean_length = collect_samples(
            samples, t, model, env, device
        )
        writer.add_scalar("env/mean_reward", mean_reward, i)
        writer.add_scalar("env/mean_length", mean_length, i)

        # calculate advantages
        for episode in episodes:
            episode.calculate(gamma, lam)

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
        total_lentropy = 0
        total_items = 0
        for _ in range(epochs):
            for batch in loader:
                # clear old gradients
                optimizer.zero_grad()

                # get batch data
                states, actions, probs, target_values, advantages = (
                    batch["states"].to(device),
                    batch["actions"].to(device),
                    batch["probs"].to(device),
                    batch["target_values"].to(device),
                    batch["advantages"].to(device),
                )

                # normalize advantages
                if normalize_advantages and advantages.shape[0] > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # predict
                pred_probs, pred_vals = model(states)

                # calculate lclip
                cur_action_probs = pred_probs[
                    torch.arange(pred_probs.shape[0]), actions
                ]
                ratio = cur_action_probs / torch.where(probs == 0, 1e-8, probs)
                # paper says we want to maximize this, so therefore just negate it
                lclip = -torch.minimum(
                    ratio * advantages, torch.clip(ratio, 1 - eps, 1 + eps) * advantages
                ).mean()

                # calculate lvf
                lvf = torch.mean((pred_vals - target_values) ** 2)

                # calculate entropy bonus
                # encourage low probabilities for the current actions
                # (log of probability is a negative number, smaller probability -> smaller log)
                lentropy = torch.log(cur_action_probs).mean()

                # backpropagate
                loss = lclip + vcf * lvf + ecf * lentropy
                loss.backward()
                if clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm_val)
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
                total_lentropy += lentropy
                if np.isnan(loss):
                    print(pred_probs)
                    print(pred_vals)
                    raise Exception("NAN Values!")
                writer.add_scalar("train/loss", loss, i)
                writer.add_scalar("train/lclip", lclip, i)
                writer.add_scalar("train/lvf", lvf, i)
                writer.add_scalar("train/lentropy", lentropy, i)
                writer.add_scalar("train/mean_ratio", ratio.mean().detach().item(), i)
                writer.add_scalar(
                    "train/mean_original_probs", probs.mean().detach().item(), i
                )
                writer.add_scalar(
                    "train/mean_new_probs", cur_action_probs.mean().detach().item(), i
                )
                writer.add_scalar(
                    "train/advantages", advantages.mean().detach().item(), i
                )
                writer.add_scalar(
                    "train/target_values", target_values.mean().detach().item(), i
                )
                i += 1
                total_items += 1
        # more logging
        prog.close()
        print(
            f"Average epoch loss {total_loss/total_items:.4f},"
            f" lclip {total_lclip/total_items:.4f},"
            f" lvf {total_lvf/total_items:.4f},"
            f" lent {total_lentropy/total_items:.4f}"
        )

    model.eval()
