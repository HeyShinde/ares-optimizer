"""
Experience replay buffer for RL agent.
"""

import random
from collections import deque, namedtuple
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save a transition."""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer) 