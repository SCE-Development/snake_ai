"""
File: trainer.py

This file contains the code for training an actor-critic
model using the PPO algorithm.
"""

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np
from gymnasium import Env
from .model import ActorCritic
from torch.utils.tensorboard.writer import SummaryWriter
from .episodes import prepare_environment, collect_samples, EpisodeDataset
from datetime import datetime
import os


def train(
    model: ActorCritic,
    optimizer: Optimizer,
    env: Env,
    iterations: int,
    t: int,
    num_envs: int,
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
    run_name: str = "",
    print_progress: bool = True,
    save_every: int = 0,
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
        num_envs (int): the number of environments to run in parallel
        samples (int): the number of samples to collect per iteration
        batch_size (int): the batch size for training
        epochs (int): the number of epochs to train per iteration
        gamma (float): the discount factor
        lam (float): the lambda parameter for GAE
        eps (float): the epsilon parameter for clipping
        vcf (float, optional): the value function coefficient. Defaults to 1.0.
        ecf (float, optional): the entropy coefficient. Defaults to 1.0.
        device (str, optional): the device to use (cpu or cuda). Defaults to "cpu".
        num_workers (int, optional): the number of workers to use. Defaults to 4.
        clip_norm (bool, optional): whether to clip the gradients. Defaults to True.
        clip_norm_val (float, optional): the value to use for the norm. Defaults to 2.0.
        normalize_advantages (bool, optional): whether to normalize the advantages. Defaults to False.
        run_name (str, optional): the name of the run. Defaults to the current datetime.
        print_progress (bool, optional): whether to print the progress to standard output. Defaults to True.
        save_every (int, optional): how often, in iterations, to save the model's weights
    """
    model.to(device)

    if not run_name:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # log hparams
    hparams_writer = SummaryWriter(log_dir="runs")
    hparams_writer.add_hparams(
        {
            "iterations": iterations,
            "t": t,
            "num_envs": num_envs,
            "samples": samples,
            "batch_size": batch_size,
            "epochs": epochs,
            "gamma": gamma,
            "lambda": lam,
            "eps": eps,
            "vcf": vcf,
            "ecf": ecf,
        },
        {},
        run_name=run_name,
    )
    hparams_writer.close()  # close hparam writer
    # and clean up extra tfevents file
    to_remove = []
    for file in os.listdir("runs"):
        if file.startswith("events.out.tfevents") and os.path.isfile(
            os.path.join("runs", file)
        ):
            to_remove.append(file)
    for file in to_remove:
        os.remove(os.path.join("runs", file))
    # and finally create actual writer
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    # prepare environment and initialize i (optimization step)
    i = 0
    env = prepare_environment(env, t, num_envs)

    for iteration in range(iterations):
        if print_progress:
            print(f"iteration {iteration+1}")
        model.eval()
        time_start = datetime.now()
        episodes, mean_reward, mean_length = collect_samples(
            samples, model, env, device
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
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            pin_memory_device=device,
        )
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
                states, actions, log_probs, target_values, advantages = (
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
                pred_log_probs, pred_vals = model(states)

                # calculate lclip
                cur_action_log_probs: torch.Tensor = pred_log_probs[
                    torch.arange(pred_log_probs.shape[0]), actions
                ]
                # a / b = e**(log(a) - log(b))
                # apparently this has higher numerical stability and will avoid the
                # nan's that we sometimes got when working with raw probs
                ratio = torch.exp(cur_action_log_probs - log_probs)
                # paper says we want to maximize this, so therefore just negate it
                lclip = -torch.minimum(
                    ratio * advantages,
                    torch.clip(ratio, 1 - eps, 1 + eps) * advantages,
                ).mean()

                # calculate lvf
                lvf = torch.mean((pred_vals - target_values) ** 2)

                # calculate entropy bonus
                # encourage low probabilities for the current actions
                # (log of probability is a negative number, smaller probability -> more negative log)
                lentropy = cur_action_log_probs.mean()

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
                total_loss += loss
                total_lclip += lclip
                total_lvf += lvf
                total_lentropy += lentropy
                if np.isnan(loss):
                    print(pred_log_probs)
                    print(pred_vals)
                    raise Exception("NAN Values!")
                writer.add_scalar("train/loss", loss, i)
                writer.add_scalar("train/lclip", lclip, i)
                writer.add_scalar("train/lvf", lvf, i)
                writer.add_scalar("train/lentropy", lentropy, i)
                writer.add_scalar("train/mean_ratio", ratio.mean().detach().item(), i)
                writer.add_scalar(
                    "train/mean_original_probs",
                    torch.exp(log_probs).mean().detach().item(),
                    i,
                )
                writer.add_scalar(
                    "train/mean_new_probs",
                    torch.exp(cur_action_log_probs).mean().detach().item(),
                    i,
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
        if print_progress:
            print(
                f"Average epoch loss {total_loss/total_items:.4f},",
                f"lclip {total_lclip/total_items:.4f},",
                f"lvf {total_lvf/total_items:.4f},",
                f"lent {total_lentropy/total_items:.4f}",
            )
        writer.add_scalar("train/iteration", iteration, i)
        if save_every != 0 and iteration != 0 and iteration % save_every == 0:
            model.cpu()
            torch.save(model.state_dict(), f"{run_name}-iteration-{iteration}.pt")
            model.to(device)

    model.eval()
