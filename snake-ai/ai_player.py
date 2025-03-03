from snake_ai.model import CustomCNNActorCritic
import gymnasium as gym
from gymnasium.wrappers.frame_stack import FrameStack
import pygame
import torch
import snake_env
from argparse import ArgumentParser

snake_env  # prevent unused warning


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "frame_stack",
        type=int,
        help="number of frames to stack; should match the number of frames the model was trained on",
    )
    parser.add_argument("model_path", type=str, help="path to the model to load")
    parser.add_argument("hidden_dim", type=int, help="hidden dimension of the model")
    parser.add_argument("--cuda", action="store_true", help="use cuda instead of cpu")
    parser.add_argument("--verbose", action="store_true", help="print debug info")
    parser.add_argument(
        "--deterministic", action="store_true", help="use deterministic actions"
    )
    args = parser.parse_args()

    frame_stack = args.frame_stack
    model_path = args.model_path
    cuda = args.cuda
    hidden_dim = args.hidden_dim
    verbose = args.verbose
    deterministic = args.deterministic

    device = "cpu" if not cuda else "cuda"
    env = gym.make("snake_env/SnakeEnv-v0", render_mode="human")

    model = CustomCNNActorCritic(
        (frame_stack, *env.observation_space.shape),
        env.action_space.n,
        hidden_dim,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)

    # stack into a frame stack
    env = FrameStack(env, frame_stack)
    obs, _ = env.reset()
    clock = pygame.time.Clock()
    good = True
    total_reward = 0
    while good:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                good = False

        if verbose:
            with torch.no_grad():
                log_probs, pred_val = model(torch.tensor(obs).unsqueeze(0).to(device))
            log_probs: torch.Tensor = log_probs.squeeze()
            if deterministic:
                action = log_probs.argmax().item()
            else:
                action = torch.exp(log_probs).multinomial(1).squeeze().item()
            print("Predicted value:", pred_val, "Probabilities:", torch.exp(log_probs))
        else:
            action = model.predict(
                torch.tensor(obs).to(device), deterministic=deterministic
            )[0]

        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if verbose:
            print("Reward:", reward)
        total_reward += reward
        if terminated or truncated:
            good = False
        clock.tick(60)

    env.close()
    print("Finished with total reward", total_reward)
