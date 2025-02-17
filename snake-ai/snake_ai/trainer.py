"""
File: trainer.py

This file contains the code for training an actor-critic
model using the PPO algorithm.
"""

import torch
from typing import Union
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
from gymnasium.vector import VectorEnv
from .model import CNNActorCritic, MLPActorCritic
from tqdm import tqdm
from torch.utils.tensorboard.writer import SummaryWriter
from .episodes import collect_samples, EpisodeDataset
from datetime import datetime


def train(
    model: Union[CNNActorCritic, MLPActorCritic],
    optimizer: Optimizer,
    env: VectorEnv,
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
        model (Union[CNNActorCritic, MLPActorCritic]): the actor-critic model to train
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
        time_start = datetime.now()
        episodes, mean_reward, mean_length = collect_samples(
            samples, t, model, env, device
        )
        time_end = datetime.now()
        writer.add_scalar("env/mean_reward", mean_reward, i)
        writer.add_scalar("env/mean_length", mean_length, i)
        writer.add_scalar(
            "env/collection_time", (time_end - time_start).total_seconds(), i
        )

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
            f"Average epoch loss {total_loss/total_items:.4f},",
            f"lclip {total_lclip/total_items:.4f},",
            f"lvf {total_lvf/total_items:.4f},",
            f"lent {total_lentropy/total_items:.4f}",
        )

    model.eval()
