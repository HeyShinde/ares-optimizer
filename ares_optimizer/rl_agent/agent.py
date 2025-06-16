"""
Deep RL agent for code optimization using DQN.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ares_optimizer.rl_agent.models import DQN
from ares_optimizer.rl_agent.replay_buffer import PrioritizedReplayBuffer, Transition
import logging
from typing import Optional, Dict, Any

class DQNAgent:
    def __init__(
        self,
        state_dim: int = 18,
        action_dim: int = 5,
        device: str = 'cpu',
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_capacity: int = 10000,
        target_update_freq: int = 100,
        target_update_tau: float = 0.005,  # Soft update parameter
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 500,
        alpha: float = 0.6,  # Priority exponent
        beta: float = 0.4,   # Importance sampling exponent
        beta_increment: float = 0.001
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.target_update_tau = target_update_tau
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.step_count = 0
        self.last_target_update = 0

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=alpha,
            beta=beta,
            beta_increment=beta_increment
        )
        self.logger = logging.getLogger(__name__)

    def select_action(self, state: np.ndarray) -> int:
        """Select an action using epsilon-greedy policy."""
        self.step_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            np.exp(-1. * self.step_count / self.epsilon_decay)
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def _soft_update_target_network(self):
        """Soft update target network using Polyak averaging."""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.target_update_tau * policy_param.data + (1.0 - self.target_update_tau) * target_param.data
            )

    def _hard_update_target_network(self):
        """Hard update target network by copying policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def update_target_network(self) -> Dict[str, Any]:
        """
        Update the target network based on the update frequency.
        Returns metrics about the update.
        """
        metrics = {
            'steps_since_update': self.step_count - self.last_target_update,
            'update_type': None,
            'network_diff': None
        }

        if self.step_count - self.last_target_update >= self.target_update_freq:
            # Calculate network difference before update
            with torch.no_grad():
                diff = 0.0
                for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    diff += torch.norm(target_param.data - policy_param.data).item()
                metrics['network_diff'] = diff

            # Perform hard update
            self._hard_update_target_network()
            self.last_target_update = self.step_count
            metrics['update_type'] = 'hard'
            self.logger.info(f"Hard target network update at step {self.step_count}")
        else:
            # Perform soft update
            self._soft_update_target_network()
            metrics['update_type'] = 'soft'

        return metrics

    def train_step(self) -> Optional[Dict[str, Any]]:
        """Perform a single training step."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample transitions with priorities
        transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*transitions)

        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Compute current Q values
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1, keepdim=True)[0]
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # Compute TD errors for priority updates
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()

        # Compute weighted loss
        loss = (weights * nn.functional.mse_loss(q_values, target_q_values, reduction='none')).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities in replay buffer
        self.replay_buffer.update_priorities(indices, td_errors)

        # Update target network and get metrics
        target_metrics = self.update_target_network()

        # Combine all metrics
        metrics = {
            'loss': loss.item(),
            'mean_td_error': np.mean(td_errors),
            'max_td_error': np.max(td_errors),
            'epsilon': self.epsilon,
            **target_metrics
        }

        return metrics

    def save(self, path: str):
        """Save the model and replay buffer."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'epsilon': self.epsilon,
            'last_target_update': self.last_target_update
        }, path)
        self.replay_buffer.save(path + '.buffer')

    def load(self, path: str):
        """Load the model and replay buffer."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        self.epsilon = checkpoint['epsilon']
        self.last_target_update = checkpoint.get('last_target_update', 0)
        self.replay_buffer.load(path + '.buffer') 