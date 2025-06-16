"""
Experience replay buffer for RL agent with prioritized sampling.
"""

import random
import numpy as np
from collections import deque, namedtuple
import logging
from typing import List, Tuple, Optional, Dict, Any
import torch

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer with efficient sampling and memory management."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6
    ):
        """
        Initialize the replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            alpha: Priority exponent (0 = uniform sampling, 1 = fully prioritized)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Rate at which beta increases to 1
            epsilon: Small value to prevent zero priorities
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
        self.logger = logging.getLogger(__name__)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Save a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
        """
        # Store transition with maximum priority
        max_priority = max(self.priorities) if self.buffer else 1.0
        self.buffer.append(Transition(state, action, reward, next_state, done))
        self.priorities.append(float(max_priority))

    def sample(self, batch_size: int) -> Tuple[Transition, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions with priorities.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (transitions, indices, weights)
        """
        if len(self.buffer) < batch_size:
            self.logger.warning(f"Buffer size ({len(self.buffer)}) smaller than batch size ({batch_size})")
            batch_size = len(self.buffer)

        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights

        # Get transitions
        transitions = [self.buffer[idx] for idx in indices]

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return Transition(*zip(*transitions)), indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for transitions.

        Args:
            indices: Indices of transitions to update
            priorities: New priorities for these transitions
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority)
            self.max_priority = max(self.max_priority, priority)

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary containing memory usage information
        """
        total_size = 0
        for transition in self.buffer:
            # Estimate size of each component
            state_size = transition.state.nbytes if isinstance(transition.state, np.ndarray) else 0
            next_state_size = transition.next_state.nbytes if isinstance(transition.next_state, np.ndarray) else 0
            total_size += state_size + next_state_size + 8  # 8 bytes for action, reward, done

        return {
            'total_size_mb': total_size / (1024 * 1024),
            'buffer_size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity
        }

    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        self.priorities.clear()
        self.max_priority = 1.0

    def __len__(self) -> int:
        """Get the current size of the buffer."""
        return len(self.buffer)

    def save(self, path: str) -> None:
        """
        Save the buffer to disk.

        Args:
            path: Path to save the buffer
        """
        try:
            data = {
                'buffer': list(self.buffer),
                'priorities': list(self.priorities),
                'max_priority': self.max_priority
            }
            torch.save(data, path)
            self.logger.info(f"Buffer saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving buffer: {str(e)}")

    def load(self, path: str) -> None:
        """
        Load the buffer from disk.

        Args:
            path: Path to load the buffer from
        """
        try:
            data = torch.load(path)
            self.buffer = deque(data['buffer'], maxlen=self.capacity)
            self.priorities = deque(data['priorities'], maxlen=self.capacity)
            self.max_priority = data['max_priority']
            self.logger.info(f"Buffer loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading buffer: {str(e)}")
            self.clear() 