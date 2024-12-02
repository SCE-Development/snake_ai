"""
Snake Eater
Made with PyGame

Implementation is credit to https://github.com/rajatdiptabiswas/snake-pygame/tree/master
"""

import pygame
import gymnasium as gym
import numpy as np
from collections import deque
from typing import Literal, Optional, Tuple, Any


class SnakeEnv(gym.Env):
    # metadata
    metadata = {"render_modes": ["human"], "render_fps": 10}

    # constants
    EMPTY = 0
    BODY = 1
    HEAD = 2
    FOOD = 3

    # game constants
    SIZE_X = 72
    SIZE_Y = 48
    OFFSET = 20  # offset for valid spawns
    ACTION_MAP = ["UP", "DOWN", "LEFT", "RIGHT"]
    DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # action space
    action_space = gym.spaces.Discrete(4)  # up, down, left, right
    # 0 is empty, 1 is body, 2 is body head, 3 is food
    observation_space = gym.spaces.Box(
        low=0, high=3, shape=(SIZE_X + 2, SIZE_Y + 2, 4), dtype=np.uint8
    )

    # display related constants
    FRAME_SCALE = 10

    def __init__(
        self,
        render_mode: Optional[Literal["human"]] = None,
    ):
        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            pygame.display.set_caption("Snake Eater")
            self.game_window = pygame.display.set_mode(
                (self.SIZE_X * self.FRAME_SCALE, self.SIZE_Y * self.FRAME_SCALE)
            )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment

        Args:
            seed (int, optional): The seed to use. Defaults to None.
            options (dict[str, Any], optional): The options to use. Defaults to None.

        Returns:
            Tuple[np.ndarray, dict[str, Any]]: The observation, and info
            - observation (np.ndarray): The observation of the game
            - info (dict[str, Any]): The info of the game
        """
        super().reset(seed=seed)
        # set the initial position of the snake
        # positions are not in frame space; they are in grid space
        # start in a random position
        pos = [
            np.random.randint(self.OFFSET, self.SIZE_X - self.OFFSET),
            np.random.randint(self.OFFSET, self.SIZE_Y - self.OFFSET),
        ]
        orientation = np.random.randint(0, 4)  # up, down, left right

        self.snake_pos = pos
        self.snake_body = deque()
        for i in range(np.random.randint(3, 5)):
            xi, yi = self.DIRECTIONS[orientation]
            self.snake_body.append([pos[0] - i * xi, pos[1] - i * yi])
        print("starting orientation", orientation, "body", self.snake_body)

        # initialize food position
        self._spawn_food()

        self.direction = self.ACTION_MAP[orientation]
        self.score = 0

        return self._get_obs(), self._get_info()

    def _spawn_food(self):
        """
        Keep trying to spawn food until it is not on the snake
        """
        self.food_pos = [
            self.np_random.integers(1, (self.SIZE_X)),
            self.np_random.integers(1, (self.SIZE_Y)),
        ]
        while self.food_pos in self.snake_body:
            self.food_pos = [
                self.np_random.integers(1, (self.SIZE_X)),
                self.np_random.integers(1, (self.SIZE_Y)),
            ]

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, bool, dict[str, Any]]:
        """
        Take a step in the game

        Args:
            action (int): The action to take. 0 is up, 1 is down, 2 is left, 3 is right

        Returns:
            Tuple[np.ndarray, int, bool, bool, dict[str, Any]]: The observation, reward, terminated, truncated, and info
            - observation (np.ndarray): The observation of the game. 0 is empty, 1 is body, 2 is body head, 3 is food
            - reward (int): The reward for the action
            - terminated (bool): Whether the agent reaches the terminal state
            - truncated (bool): Whether or not the game has been truncated (out of time)
            - info (dict): Additional information about the game
        """
        # 0 is up, 1 is down, 2 is left, 3 is right
        # Making sure the snake cannot move in the opposite direction instantaneously
        change_to = self.ACTION_MAP[action]

        if change_to == "UP" and self.direction != "DOWN":
            self.direction = "UP"
        if change_to == "DOWN" and self.direction != "UP":
            self.direction = "DOWN"
        if change_to == "LEFT" and self.direction != "RIGHT":
            self.direction = "LEFT"
        if change_to == "RIGHT" and self.direction != "LEFT":
            self.direction = "RIGHT"

        # Moving the snake
        if self.direction == "UP":
            self.snake_pos[1] -= 1
        if self.direction == "DOWN":
            self.snake_pos[1] += 1
        if self.direction == "LEFT":
            self.snake_pos[0] -= 1
        if self.direction == "RIGHT":
            self.snake_pos[0] += 1

        # Snake body growing mechanism
        self.snake_body.appendleft(list(self.snake_pos))
        dist = abs(self.snake_pos[0] - self.food_pos[0]) + abs(
            self.snake_pos[1] - self.food_pos[1]
        )
        reward = 1 / (dist + 1)
        if (
            self.snake_pos[0] == self.food_pos[0]
            and self.snake_pos[1] == self.food_pos[1]
        ):
            self.score += 1
            reward += 1
            self._spawn_food()
        else:
            self.snake_body.pop()

        # Game Over conditions
        terminated = False
        truncated = False
        # Getting out of bounds
        if self.snake_pos[0] < 0 or self.snake_pos[0] >= self.SIZE_X:
            terminated = True
        if self.snake_pos[1] < 0 or self.snake_pos[1] >= self.SIZE_Y:
            terminated = True
        # hitting itself
        terminated = terminated or self.snake_pos in list(self.snake_body)[1:]
        if terminated:
            reward = -1

        # return the observation, reward, terminated, truncated, and info
        observation = self._get_obs()
        info = self._get_info()
        return (observation, reward, terminated, truncated, info)

    def _get_obs(self):
        # +2 for walls
        observation = np.zeros((self.SIZE_X + 2, self.SIZE_Y + 2, 4), dtype=np.uint8)
        observation[self.snake_pos[0] + 1, self.snake_pos[1] + 1, self.HEAD] = 1
        for x, y in list(self.snake_body)[1:]:
            observation[x + 1, y + 1, self.BODY] = 1
        observation[self.food_pos[0] + 1, self.food_pos[1] + 1, self.FOOD] = 1
        return observation

    def _get_info(self):
        return {"score": self.score}

    def render(self):
        if self.render_mode == "human":
            white = pygame.Color(255, 255, 255)
            black = pygame.Color(0, 0, 0)
            self.game_window.fill(black)
            self._show_score(1, white, "consolas", 20)
            self._draw_snake()
            self._draw_food()
            pygame.display.update()

    def _show_score(self, choice, color, font, size):
        score_font = pygame.font.SysFont(font, size)
        score_surface = score_font.render(f"Score : {self.score}", True, color)
        score_rect = score_surface.get_rect()
        if choice == 1:
            score_rect.midtop = (self.SIZE_X, 15)
        else:
            score_rect.midtop = (
                self.SIZE_X * self.FRAME_SCALE / 2,
                self.SIZE_Y * self.FRAME_SCALE / 1.25,
            )
        self.game_window.blit(score_surface, score_rect)

    def _draw_snake(self):
        green = pygame.Color(0, 255, 0)
        for pos in self.snake_body:
            pygame.draw.rect(
                self.game_window,
                green,
                pygame.Rect(
                    pos[0] * self.FRAME_SCALE,
                    pos[1] * self.FRAME_SCALE,
                    self.FRAME_SCALE,
                    self.FRAME_SCALE,
                ),
            )

    def _draw_food(self):
        white = pygame.Color(255, 255, 255)
        pygame.draw.rect(
            self.game_window,
            white,
            pygame.Rect(
                self.food_pos[0] * self.FRAME_SCALE,
                self.food_pos[1] * self.FRAME_SCALE,
                self.FRAME_SCALE,
                self.FRAME_SCALE,
            ),
        )

    def close(self):
        if self.render_mode == "human":
            pygame.quit()
