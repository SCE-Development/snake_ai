import snake_env
import numpy as np
import gymnasium as gym
import pygame

snake_env  # prevent unused

env = gym.make("snake_env/SnakeEnv-v0", render_mode="human")
env.reset()
clock = pygame.time.Clock()
while True:
    action = np.random.randint(4)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break
    clock.tick(25)

env.close()
