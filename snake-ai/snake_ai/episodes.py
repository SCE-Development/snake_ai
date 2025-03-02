from typing import List, Tuple
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from bisect import bisect_right
from .model import ActorCritic
from gymnasium.vector import VectorEnv, SyncVectorEnv
from gymnasium import Env
from gymnasium.wrappers.time_limit import TimeLimit
import copy
import numpy as np
import torch


@dataclass
class EpisodeRecord:
    states: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    actions: torch.Tensor
    probs: torch.Tensor
    last_value: float
    terminated: bool
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
        # terminated -> actually end
        # otherwise, truncated, so pretend it didn't end
        deltas.append(
            self.rewards[self.T - 1]
            - self.values[t]
            + (0.0 if self.terminated else self.last_value)
        )
        assert len(deltas) == self.T

        # instead of doing the N^2 version of calculating advantages,
        # go from the back instead
        advantages = torch.zeros((len(deltas),))
        # set the last item
        advantages[-1] = deltas[-1]
        for t in range(self.T - 2, -1, -1):
            advantages[t] = deltas[t] + (lam * gamma) * advantages[t + 1]
        self.advantages = advantages
        assert self.advantages.shape[0] == self.T

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
            cur += record.T
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


def prepare_environment(env: Env, t: int, num_envs: int):
    """
    Returns a vectorized environment where each
    individual environment is limited to at most t samples

    Arguments:
        env (Env): the environment to wrap
        t (int): the maximum number of samples to collect before resetting the env
        num_envs (int): the number of environments to create

    Returns:
        env (VectorEnv): the vectorized environment
    """

    def make_env():
        c = copy.deepcopy(env)
        return TimeLimit(c, t)

    return SyncVectorEnv([make_env for _ in range(num_envs)])


def collect_samples(
    samples: int,
    model: ActorCritic,
    env: VectorEnv,
    device: str,
) -> Tuple[List[EpisodeRecord], float, float]:
    """
    Collects `samples` samples from the environment, resetting it at first
    Assumes that model.get_value and model.get_action have gradients

    Arguments:
        samples (int): the number of samples to collect
        model (ActorCritic): the actor-critic
        env (Env): the environment to use
        device (str): the device to use

    Returns:
        ret (Tuple[List[EpisodeRecord], float, float): a tuple containing
            the list of episode records, the mean reward, and the mean
            episode length
    """
    cur_states = [[] for _ in range(env.num_envs)]
    cur_values = [[] for _ in range(env.num_envs)]
    cur_rewards = [[] for _ in range(env.num_envs)]
    cur_actions = [[] for _ in range(env.num_envs)]
    cur_probs = [[] for _ in range(env.num_envs)]
    records = []

    obs, _ = env.reset()
    cur_length = [0 for _ in range(env.num_envs)]
    cur_reward = [0 for _ in range(env.num_envs)]

    logged_samples = 0
    total_reward = 0
    total_length = 0
    while logged_samples < samples:
        # take current and prepare to step
        for e in range(env.num_envs):
            cur_states[e].append(obs[e])
        actions, log_probs, values = model.predict_batched(
            torch.tensor(obs).to(device), deterministic=False
        )
        actions = actions.cpu().numpy()
        log_probs = log_probs.cpu().numpy()
        values = values.cpu().numpy()

        for e in range(env.num_envs):
            cur_values[e].append(values[e])
            cur_actions[e].append(actions[e])
            cur_probs[e].append(log_probs[e])

        # step in the environment
        obs, reward, terminated, truncated, info = env.step(actions)
        for e in range(env.num_envs):
            cur_rewards[e].append(reward[e])
            cur_reward[e] += reward[e]

        for e in range(env.num_envs):
            if terminated[e] or truncated[e]:
                assert cur_length[e] > 2
                logged_samples += cur_length[e]

                final_value = 0.0
                if truncated[e]:
                    # calculate final value by using model
                    next_obs = info["final_observation"][e]
                    _, _, final_value = model.predict(torch.tensor(next_obs).to(device))

                # new episode
                records.append(
                    EpisodeRecord(
                        states=torch.from_numpy(np.array(cur_states[e])),
                        values=torch.tensor(cur_values[e]),
                        rewards=torch.tensor(cur_rewards[e]),
                        actions=torch.tensor(cur_actions[e]),
                        probs=torch.tensor(cur_probs[e]),
                        last_value=final_value,
                        terminated=terminated[e],
                    )
                )
                total_reward += cur_reward[e]
                total_length += cur_length[e]

                # environments autoreset, so simply take care of these
                cur_rewards[e] = []
                cur_states[e] = []
                cur_values[e] = []
                cur_length[e] = 0
                cur_reward[e] = 0
            cur_length[e] += 1

    # in case we have extra remaining that we didn't include in the list yet
    # for e in range(env.num_envs):
    #     if cur_length[e] > 2:
    #         # new episode
    #         records.append(
    #             EpisodeRecord(
    #                 states=torch.tensor(np.array(cur_states[e])),
    #                 values=torch.tensor(cur_values[e]),
    #                 rewards=torch.tensor(cur_rewards[e]),
    #                 actions=torch.tensor(cur_actions[e]),
    #                 probs=torch.tensor(cur_probs[e]),
    #             )
    #         )
    #         total_reward += cur_reward[e]
    #         total_length += cur_length[e]
    return records, total_reward / len(records), total_length / len(records)
