"""
Performance tests for the RL agent.
"""

import pytest
import numpy as np
import torch
import time
from ares_optimizer.rl_agent.agent import DQNAgent
import logging
import os
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def agent():
    """Create a DQN agent for testing."""
    return DQNAgent(
        state_dim=18,
        action_dim=5,
        device='cpu',
        batch_size=32,
        buffer_capacity=1000
    )

def test_training_speed(agent):
    """Test the training speed of the agent."""
    # Generate random transitions
    num_transitions = 1000
    states = np.random.randn(num_transitions, 18)
    actions = np.random.randint(0, 5, num_transitions)
    rewards = np.random.randn(num_transitions)
    next_states = np.random.randn(num_transitions, 18)
    dones = np.random.randint(0, 2, num_transitions)

    # Store transitions
    for i in range(num_transitions):
        agent.store_transition(
            states[i],
            actions[i],
            rewards[i],
            next_states[i],
            dones[i]
        )

    # Measure training time
    start_time = time.time()
    num_steps = 100
    losses = []
    
    for _ in range(num_steps):
        metrics = agent.train_step()
        if metrics:
            losses.append(metrics['loss'])

    end_time = time.time()
    training_time = end_time - start_time

    # Verify performance
    assert training_time < 1.0, f"Training too slow: {training_time:.2f}s"
    assert len(losses) > 0, "No training steps completed"
    assert all(not np.isnan(loss) for loss in losses), "NaN losses detected"

def test_memory_usage(agent):
    """Test memory usage during training."""
    # Generate random transitions
    num_transitions = 1000
    states = np.random.randn(num_transitions, 18)
    actions = np.random.randint(0, 5, num_transitions)
    rewards = np.random.randn(num_transitions)
    next_states = np.random.randn(num_transitions, 18)
    dones = np.random.randint(0, 2, num_transitions)

    # Store transitions
    for i in range(num_transitions):
        agent.store_transition(
            states[i],
            actions[i],
            rewards[i],
            next_states[i],
            dones[i]
        )

    # Get memory usage before training
    buffer_metrics = agent.replay_buffer.get_memory_usage()
    initial_memory = buffer_metrics['total_size_mb']

    # Train for some steps
    for _ in range(100):
        agent.train_step()

    # Get memory usage after training
    buffer_metrics = agent.replay_buffer.get_memory_usage()
    final_memory = buffer_metrics['total_size_mb']

    # Verify memory usage
    assert final_memory >= initial_memory, "Memory usage should not decrease"
    assert final_memory < 100, f"Memory usage too high: {final_memory:.2f}MB"

def test_convergence(agent):
    """Test the convergence of the agent."""
    # Create a simple environment
    def get_reward(state, action):
        return 1.0 if action == np.argmax(state) else -1.0

    # Train the agent
    num_episodes = 100
    rewards = []
    losses = []

    for episode in range(num_episodes):
        state = np.random.randn(18)
        action = agent.select_action(state)
        reward = get_reward(state, action)
        next_state = np.random.randn(18)
        done = episode == num_episodes - 1

        agent.store_transition(state, action, reward, next_state, done)
        metrics = agent.train_step()
        
        if metrics:
            losses.append(metrics['loss'])
        rewards.append(reward)

    # Verify convergence
    assert len(losses) > 0, "No training steps completed"
    assert np.mean(losses[-10:]) < np.mean(losses[:10]), "Loss should decrease"
    assert np.mean(rewards[-10:]) > np.mean(rewards[:10]), "Rewards should increase"

def test_target_network_updates(agent):
    """Test target network update mechanism."""
    # Store initial target network state
    initial_target_state = {
        name: param.clone()
        for name, param in agent.target_net.state_dict().items()
    }

    # Train for some steps
    num_steps = 200  # More than target_update_freq
    for _ in range(num_steps):
        state = np.random.randn(18)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(18)
        done = False

        agent.store_transition(state, action, reward, next_state, done)
        metrics = agent.train_step()

    # Verify target network updates
    final_target_state = agent.target_net.state_dict()
    differences = []
    
    for name in initial_target_state:
        diff = torch.norm(
            initial_target_state[name] - final_target_state[name]
        ).item()
        differences.append(diff)

    assert any(diff > 0 for diff in differences), "Target network should be updated"
    assert agent.last_target_update > 0, "Target network update counter should be incremented"

def test_exploration_exploitation(agent):
    """Test exploration vs exploitation behavior."""
    # Track action selection
    num_steps = 1000
    actions = []
    epsilons = []

    for _ in range(num_steps):
        state = np.random.randn(18)
        action = agent.select_action(state)
        actions.append(action)
        epsilons.append(agent.epsilon)

    # Verify exploration
    unique_actions = len(set(actions))
    assert unique_actions > 1, "Agent should explore different actions"

    # Verify epsilon decay
    assert epsilons[-1] < epsilons[0], "Epsilon should decrease over time"
    assert epsilons[-1] >= agent.epsilon_end, "Epsilon should not go below minimum"

def test_batch_processing(agent):
    """Test batch processing efficiency."""
    # Generate random transitions
    num_transitions = 1000
    states = np.random.randn(num_transitions, 18)
    actions = np.random.randint(0, 5, num_transitions)
    rewards = np.random.randn(num_transitions)
    next_states = np.random.randn(num_transitions, 18)
    dones = np.random.randint(0, 2, num_transitions)

    # Store transitions
    for i in range(num_transitions):
        agent.store_transition(
            states[i],
            actions[i],
            rewards[i],
            next_states[i],
            dones[i]
        )

    # Measure batch processing time
    batch_sizes = [16, 32, 64, 128]
    processing_times = []

    for batch_size in batch_sizes:
        agent.batch_size = batch_size
        start_time = time.time()
        
        for _ in range(10):  # Process 10 batches
            agent.train_step()
            
        end_time = time.time()
        processing_times.append(end_time - start_time)

    # Verify batch processing
    assert all(time > 0 for time in processing_times), "Processing times should be positive"
    assert all(time < 1.0 for time in processing_times), "Processing too slow" 