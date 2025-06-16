"""
Deep RL agent for code optimization using DQN.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ares_optimizer.rl_agent.models import DQN
from ares_optimizer.rl_agent.replay_buffer import ReplayBuffer, Transition

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
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 500
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.step_count = 0

        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

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
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*transitions)

        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        # Q(s, a)
        q_values = self.policy_net(state_batch).gather(1, action_batch)
        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1, keepdim=True)[0]
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        loss = nn.functional.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically update target network
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict()) 