import torch
from gymnasium import Env
from typing import List
from .model import PPOModel
from dataclasses import dataclass, field


@dataclass
class EpisodeRecord:
    states: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor = field(init=False)
    T: int

    def __post_init__(self):
        self.T = self.states.shape[0]


def collect_samples(samples: int, model: PPOModel, env: Env) -> List[EpisodeRecord]:
    """
    Collects `samples` samples from the environment, resetting it at first
    Assumes that model.get_value and model.get_action have gradients

    Arguments:
        samples (int): the number of samples to collect
        model (PPOModel): the actor-critic
        env (Env): the environment to use

    Returns:
        List[EpisodeRecord] - the states, values, and rewards
    """
    cur_states = []
    cur_values = []
    cur_rewards = []
    records = []

    obs = env.reset()
    num_samples = 0
    while num_samples < samples:
        # step in the environment
        cur_states.append(obs)
        cur_values.append(model.get_value(obs))
        action = model.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        cur_rewards.append(reward)

        if terminated or truncated:
            # new episode
            records.append(
                EpisodeRecord(
                    states=torch.tensor(cur_states),
                    values=torch.tensor(cur_values),
                    rewards=torch.tensor(cur_rewards),
                )
            )
            cur_rewards = []
            cur_states = []
            cur_values = []
        num_samples += 1

    # in case we have extra
    if len(cur_values) > 0:
        # new episode
        records.append(
            EpisodeRecord(
                states=torch.tensor(cur_states),
                values=torch.tensor(cur_values),
                rewards=torch.tensor(cur_rewards),
                advantages=torch.zeros((len(cur_states) - 1,)),
            )
        )
    return records


def train(
    model: PPOModel,
    env: Env,
    iterations: int,
    samples: int,
    k: int,
    m: int,
    gamma: float,
    lam: float,
    eps: float,
):
    # see page 5 of https://arxiv.org/pdf/1707.06347
    for _ in range(iterations):
        episodes = collect_samples(samples, model, env)

        # compute advantages for all episodes
        lclip = torch.tensor(0, dtype=torch.float32)
        for episode in episodes:
            deltas = []
            for t in range(episode.T - 1):
                deltas.append(
                    episode.rewards[t]
                    + gamma * episode.values[t + 1]
                    - episode.values[t]
                )
            # instead of doing the N^2 version of calculating advantages,
            # go from the back instead
            advantages = torch.zeros_like(deltas)
            advantages[episode.T - 2] = deltas[episode.T - 2]  # -2 since 0-indexed
            for t in range(episode.T - 3, -1, -1):
                advantages[t] = deltas[t] + (lam * gamma) * advantages[t + 1]
            episode.advantages = advantages

            lclip += torch.min(
                episode.rewards[: episode.T] * advantages,
                torch.clip(episode.rewards[: episode.T], 1 - eps, 1 + eps) * advantages,
            )
        # optimize
        lclip.backward()
        # TODO - rework this
