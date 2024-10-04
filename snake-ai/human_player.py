import snake_env
import gymnasium as gym
import pygame

snake_env  # prevent unused

env = gym.make("snake_env/SnakeEnv-v0", render_mode="human")
env.reset()
clock = pygame.time.Clock()
action = 2
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = 0
            elif event.key == pygame.K_DOWN:
                action = 1
            elif event.key == pygame.K_LEFT:
                action = 2
            elif event.key == pygame.K_RIGHT:
                action = 3
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break
    clock.tick(25)

env.close()
