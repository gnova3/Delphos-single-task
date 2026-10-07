"""Q-network and bounded experience replay."""
import random
from collections import deque
from typing import Any, List, Tuple
import torch.nn as nn

class DQNetwork(nn.Module):
    """
    Deep Q-Network with a variable number of hidden layers.

    Args:
        input_size (int): Size of the input tensor.
        output_size (int): Size of the output tensor.
        hidden_layers (List[int], optional): List of hidden layer sizes. Defaults to (128, 64).
    """
    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int] = (128, 64)) -> None:
        super().__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ReplayBuffer:
    """
    Replay buffer for storing and sampling past experiences.

    Args:
        max_size (int): Maximum number of experiences to store.
    """
    def __init__(self, max_size: int) -> None:
        self.buffer = deque(maxlen=max_size)

    def add(self, transition: Tuple) -> None:
        """
        Add a new experience to the buffer.

        Args:
            transition (Tuple): Experience tuple to add.
        """
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Any:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            Any: Batch of sampled experiences, or None if not enough data.
        """
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
