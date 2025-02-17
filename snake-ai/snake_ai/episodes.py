from typing import List, Tuple
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from bisect import bisect_right
from .model import ActorCritic
from gymnasium.vector import VectorEnv
import numpy as np
import torch


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
    samples: int,
    t: int,
    model: ActorCritic,
    env: VectorEnv,
    device: str,
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
        action, prob, value = model.predict_batched(
            torch.tensor(obs).to(device), deterministic=False
        )
        action = action.cpu().numpy()
        prob = prob.cpu().numpy()
        value = value.cpu().numpy()

        for e in range(env.num_envs):
            cur_values[e].append(value[e])
            cur_actions[e].append(action[e])
            cur_probs[e].append(prob[e])

        # step in the environment
        obs, reward, terminated, truncated, _ = env.step(action)
        for e in range(env.num_envs):
            cur_rewards[e].append(reward[e])
            cur_reward[e] += reward[e]

        for e in range(env.num_envs):
            if terminated[e] or truncated[e]:
                logged_samples += cur_length[e]
                if cur_length[e] > 2:
                    # new episode
                    records.append(
                        EpisodeRecord(
                            states=torch.from_numpy(np.array(cur_states[e])),
                            values=torch.tensor(cur_values[e]),
                            rewards=torch.tensor(cur_rewards[e]),
                            actions=torch.tensor(cur_actions[e]),
                            probs=torch.tensor(cur_probs[e]),
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
    for e in range(env.num_envs):
        if cur_length[e] > 2:
            # new episode
            records.append(
                EpisodeRecord(
                    states=torch.tensor(np.array(cur_states[e])),
                    values=torch.tensor(cur_values[e]),
                    rewards=torch.tensor(cur_rewards[e]),
                    actions=torch.tensor(cur_actions[e]),
                    probs=torch.tensor(cur_probs[e]),
                )
            )
            total_reward += cur_reward[e]
            total_length += cur_length[e]
    return records, total_reward / len(records), total_length / len(records)
