import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import asyncio
import logging
import os
from collections import deque

logger = logging.getLogger(__name__)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(
        self,
        game,
        learning_rate=0.001,
        discount_factor=0.9,
        exploration_rate=1.0,
        exploration_decay=0.999,
        min_exploration_rate=0.01,
        model_path="dqn_model.pt",
        device="cpu",
        batch_size=64,
        replay_size=10000,
        target_update=1000
    ):
        self.game = game
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate
        self.num_actions = len(game.actions)
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.replay_size = replay_size
        self.target_update = target_update
        self.steps = 0

        # Initialize DQN models
        self.model = DQN(input_dim=2, output_dim=self.num_actions).to(device)  # Input: relative coordinates
        self.target_model = DQN(input_dim=2, output_dim=self.num_actions).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = deque(maxlen=replay_size)

        # Load saved model if it exists
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                self.target_model.load_state_dict(self.model.state_dict())
                logger.info(f"Loaded DQN model from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model from {self.model_path}: {e}")
                logger.info("Starting with fresh DQN model")
        else:
            logger.info("No saved model found, starting with fresh DQN model")

    def save_model(self):
        try:
            torch.save(self.model.state_dict(), self.model_path)
            logger.info(f"Saved DQN model to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model to {self.model_path}: {e}")

    def get_state_tensor(self, state):
        # Convert state to relative coordinates: [goal_x - agent_x, goal_y - agent_y]
        goal_pos, agent_pos = state
        relative = np.array([goal_pos[0] - agent_pos[0], goal_pos[1] - agent_pos[1]], dtype=np.float32)
        return torch.tensor(relative, device=self.device, dtype=torch.float32)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.game.actions)  # Explore
        else:
            state_tensor = self.get_state_tensor(state)
            with torch.no_grad():
                q_values = self.model(state_tensor)
            return int(torch.argmax(q_values).item())  # Exploit

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = unzip(batch)

        states = torch.stack([self.get_state_tensor(s) for s in states]).to(self.device)
        actions = torch.tensor(actions, device=self.device, dtype=torch.long)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_states = torch.stack([self.get_state_tensor(s) for s in next_states]).to(self.device)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32)

        # Current Q-values
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss and update model
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    async def train(self, episodes=1000, save_interval=100, visualize_interval=100, start_visualize_after=0):
        recent_rewards = []
        for episode in range(episodes):
            state = self.game.reset()
            done = False
            total_reward = 0
            visualize = (episode % visualize_interval == 0 and episode >= start_visualize_after)

            logger.info(f"Starting episode {episode}{' (visualizing)' if visualize else ''}")
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.game.step(action)
                self.store_transition(state, action, reward, next_state, done)
                self.update()
                state = next_state
                total_reward += reward
                if visualize:
                    self.game.render()
                    await asyncio.sleep(1.0 / self.game.fps)

            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            recent_rewards.append(total_reward)

            # Log episode metrics
            avg_q_value = torch.mean(self.model(torch.randn(1, 2, device=self.device))).item()  # Sample Q-value
            logger.info(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.3f}, Avg Q-value: {avg_q_value:.3f}")

            # Log summary every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward = sum(recent_rewards[-100:]) / min(len(recent_rewards), 100)
                logger.info(f"Summary (Episodes {episode-99 if episode >= 99 else 0}-{episode}): Avg Reward: {avg_reward:.3f}, Epsilon: {self.epsilon:.3f}, Avg Q-value: {avg_q_value:.3f}")
                recent_rewards = []

            # Save model periodically
            if (episode + 1) % save_interval == 0:
                self.save_model()

        self.game.close()

    async def play(self):
        state = self.game.reset()
        done = False
        logger.info("Starting game...")
        self.game.render()

        while not done:
            action = self.choose_action(state)
            state, reward, done = self.game.step(action)
            self.game.render()
            if done:
                logger.info("Goal reached!")
            await asyncio.sleep(1.0 / self.game.fps)

        self.game.close()

def unzip(batch):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for s, a, r, ns, d in batch:
        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(ns)
        dones.append(d)
    return states, actions, rewards, next_states, dones