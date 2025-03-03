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
    WALL = 4
    OVERLAP = 5

    # game constants
    SIZE_X = 72
    SIZE_Y = 48
    OFFSET = 20  # offset for valid spawns
    ACTION_MAP = ["UP", "DOWN", "LEFT", "RIGHT"]
    DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    # action space
    action_space = gym.spaces.Discrete(4)  # up, down, left, right
    observation_space = gym.spaces.Box(
        low=0, high=1, shape=(SIZE_X + 2, SIZE_Y + 2, 6), dtype=np.uint8
    )
    REWARD_SCALE = 0
    EAT_REWARD = 1
    DEATH_PENALTY = 1
    INVALID_PENALTY = 0

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

        # initialize food position
        self._spawn_food()

        self.direction = self.ACTION_MAP[orientation]
        self.score = 0
        self.delta_score = 0

        return self._get_obs(), self._get_info()

    def _spawn_food(self):
        """
        Keep trying to spawn food until it is not on the snake
        """
        # TODO - what if all those spaces are taken?
        self.food_pos = [
            self.np_random.integers(self.OFFSET, self.SIZE_X - self.OFFSET),
            self.np_random.integers(self.OFFSET, self.SIZE_Y - self.OFFSET),
        ]
        while self.food_pos in self.snake_body:
            self.food_pos = [
                self.np_random.integers(self.OFFSET, self.SIZE_X - self.OFFSET),
                self.np_random.integers(self.OFFSET, self.SIZE_Y - self.OFFSET),
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

        reward = 0
        if change_to == "UP":
            if self.direction != "DOWN":
                self.direction = "UP"
            else:
                reward -= self.INVALID_PENALTY
        if change_to == "DOWN":
            if self.direction != "UP":
                self.direction = "DOWN"
            else:
                reward -= self.INVALID_PENALTY
        if change_to == "LEFT":
            if self.direction != "RIGHT":
                self.direction = "LEFT"
            else:
                reward -= self.INVALID_PENALTY
        if change_to == "RIGHT":
            if self.direction != "LEFT":
                self.direction = "RIGHT"
            else:
                reward -= self.INVALID_PENALTY

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
        if reward >= 0:
            reward += self.REWARD_SCALE * (self.SIZE_X + self.SIZE_Y - dist)
        if (
            self.snake_pos[0] == self.food_pos[0]
            and self.snake_pos[1] == self.food_pos[1]
        ):
            self.score += 1
            self.delta_score = 1
            if reward >= 0:
                reward += self.EAT_REWARD
            self._spawn_food()
        else:
            self.delta_score = 0
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
            reward = -self.DEATH_PENALTY

        # return the observation, reward, terminated, truncated, and info
        observation = self._get_obs()
        info = self._get_info()
        return (observation, reward, terminated, truncated, info)

    def _get_obs(self):
        # +2 for walls
        # zeros because that's self.EMPTY
        observation = np.zeros((self.SIZE_X + 2, self.SIZE_Y + 2), dtype=np.uint8)
        # boundaries
        observation[:, 0] = self.WALL
        observation[:, -1] = self.WALL
        observation[0, :] = self.WALL
        observation[-1, :] = self.WALL

        # head
        if observation[self.snake_pos[0] + 1, self.snake_pos[1] + 1] != self.WALL:
            observation[self.snake_pos[0] + 1, self.snake_pos[1] + 1] = self.HEAD
        else:
            observation[self.snake_pos[0] + 1, self.snake_pos[1] + 1] = self.OVERLAP
        # body and food
        for x, y in list(self.snake_body)[1:]:
            if observation[x + 1, y + 1] == self.HEAD:
                observation[x + 1, y + 1] = self.OVERLAP
            observation[x + 1, y + 1] = self.BODY
        observation[self.food_pos[0] + 1, self.food_pos[1] + 1] = self.FOOD
        return np.eye(6, dtype=np.uint8)[observation]

    def _get_info(self):
        return {"score": self.score, "delta_score": self.delta_score}

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
