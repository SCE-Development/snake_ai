from gymnasium.envs.registration import register

register(
    id="snake_env/SnakeEnv-v0",
    entry_point="snake_env.snake:SnakeEnv",
    max_episode_steps=1000,
)
