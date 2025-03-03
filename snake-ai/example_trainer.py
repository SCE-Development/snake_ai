"""
File: example_trainer.py

This file contains an example of how to train a model for the snake game.
Notice that the model used is the CustomCNNActorCritic model,
which is made specifically for the snake game.
"""

import snake_env
import gymnasium as gym
from snake_ai.trainer import train
from snake_ai.model import CustomCNNActorCritic
from torch.optim import Adam
import torch

snake_env  # prevent unused

if __name__ == "__main__":
    env = gym.make("snake_env/SnakeEnv-v0")
    frame_stack = 3
    model = CustomCNNActorCritic(
        (frame_stack, *env.observation_space.shape), env.action_space.n, 512
    )
    optim = Adam(model.parameters(), 3e-4)

    train(
        model=model,
        optimizer=optim,
        iterations=2048,
        t=2048,
        num_envs=16,
        samples=2048 * 8,
        num_stack=frame_stack,
        batch_size=512,
        epochs=6,
        gamma=0.99,
        lam=0.95,
        eps=0.25,
        ecf=0,
        vcf=1,
        clip_norm_val=0.5,
        device="cuda",
        normalize_advantages=False,
        num_workers=12,
        print_progress=False,
        clip_norm=False,
        run_name="",
        save_every=32,
    )

    torch.save(model.state_dict(), "model.pt")
