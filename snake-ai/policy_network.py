import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    def __init__(self, in_shape, output_size):
        super(FeedForwardNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_shape[0], out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        conv_output_size = in_shape[1] * in_shape[2] * 64  # Assuming the input shape (channel, width, height)
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)

        # Policy head: outputs probabilities over actions
        self.policy_head = nn.Linear(512, output_size)
        # Value head: outputs the state value (for critic in PPO)
        self.value_head = nn.Linear(512, 1)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)


        # Forward pass through convolutional layers
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value