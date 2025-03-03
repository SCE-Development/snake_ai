from dataclasses import dataclass, field
from .model import ActorCritic
from gymnasium.vector import VectorEnv, SyncVectorEnv
from gymnasium import Env
from gymnasium.wrappers.time_limit import TimeLimit
from gymnasium.wrappers.frame_stack import FrameStack
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


@dataclass
class Samples:
    episodes: list[EpisodeRecord]
    mean_reward: float
    mean_length: float
    mean_score: float


def prepare_environment(env: Env, t: int, num_envs: int, num_stack: int):
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
        c = FrameStack(c, num_stack)
        return TimeLimit(c, t)

    return SyncVectorEnv([make_env for _ in range(num_envs)])


def collect_samples(
    samples: int,
    model: ActorCritic,
    env: VectorEnv,
    device: str,
) -> Samples:
    """
    Collects `samples` samples from the environment, resetting it at first
    Assumes that model.get_value and model.get_action have gradients

    Arguments:
        samples (int): the number of samples to collect
        model (ActorCritic): the actor-critic
        env (Env): the environment to use
        device (str): the device to use

    Returns:
        ret (Samples): the collected samples
    """
    cur_states = [[] for _ in range(env.num_envs)]
    cur_values = [[] for _ in range(env.num_envs)]
    cur_rewards = [[] for _ in range(env.num_envs)]
    cur_actions = [[] for _ in range(env.num_envs)]
    cur_probs = [[] for _ in range(env.num_envs)]
    records = []

    obs, _ = env.reset()
    cur_length = [0 for _ in range(env.num_envs)]

    total_reward = 0
    total_length = 0
    total_score = 0
    while total_length < samples:
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
            total_reward += reward[e]
            total_score += info["delta_score"][e]

        for e in range(env.num_envs):
            if terminated[e] or truncated[e]:
                assert cur_length[e] > 1

                final_value = 0.0
                if truncated[e]:
                    # calculate final value by using model
                    next_obs = info["final_observation"][e]
                    _, _, final_value = model.predict(
                        torch.tensor(np.array(next_obs)).to(device)
                    )

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
                total_length += cur_length[e]

                # environments autoreset, so simply take care of these
                cur_states[e] = []
                cur_values[e] = []
                cur_rewards[e] = []
                cur_actions[e] = []
                cur_probs[e] = []
                cur_length[e] = 0
            cur_length[e] += 1

    return Samples(
        records,
        total_reward / len(records),
        total_length / len(records),
        total_score / len(records),
    )
