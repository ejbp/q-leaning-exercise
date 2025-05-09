import torch
import random
import asyncio
import logging
import sys

logger = logging.getLogger(__name__)

class QLearningAgent:
    def __init__(self, game, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.999, min_exploration_rate=0.01, model_path="q_table.pt", device="cpu"):
        self.game = game
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration_rate
        self.num_actions = len(game.actions)
        self.size = game.size
        self.model_path = model_path
        self.device = device

        # Initialize Q-table as a tensor on device
        self.q_table = torch.zeros((self.size, self.size, self.size, self.size, self.num_actions), device=device, dtype=torch.float32)
        self.actions = torch.tensor(game.actions, device=device, dtype=torch.long)

        # Load saved model if it exists
        self.load_model()

    def load_model(self):
        import os
        if os.path.exists(self.model_path):
            try:
                self.q_table = torch.load(self.model_path, map_location=self.device)
                logger.info(f"Loaded Q-table from {self.model_path}")
            except Exception as e:
                logger.error(f"Error loading model from {self.model_path}: {e}")
                logger.info("Starting with fresh Q-table")
        else:
            logger.info("No saved model found, starting with fresh Q-table")

    def save_model(self):
        try:
            torch.save(self.q_table, self.model_path)
            logger.info(f"Saved Q-table to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model to {self.model_path}: {e}")

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.game.actions)  # Explore (CPU for simplicity)
        else:
            state_idx = (state[0][0], state[0][1], state[1][0], state[1][1])
            q_values = self.q_table[state_idx[0], state_idx[1], state_idx[2], state_idx[3]]  # Get Q-values for state
            return int(torch.argmax(q_values).item())  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        state_idx = (state[0][0], state[0][1], state[1][0], state[1][1])
        next_state_idx = (next_state[0][0], next_state[0][1], next_state[1][0], next_state[1][1])

        # Convert state and action to tensors
        action = torch.tensor(action, device=self.device, dtype=torch.long)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)

        # Current Q-value
        current_q = self.q_table[state_idx[0], state_idx[1], state_idx[2], state_idx[3], action]

        # Next Q-value
        next_max_q = torch.max(self.q_table[next_state_idx[0], next_state_idx[1], next_state_idx[2], next_state_idx[3]])

        # Q-learning update
        new_q = current_q + self.lr * (reward + self.gamma * next_max_q - current_q)

        # Update Q-table
        self.q_table[state_idx[0], state_idx[1], state_idx[2], state_idx[3], action] = new_q

        
    async def train(self, episodes=1000, save_interval=100, visualize_interval=100, start_visualize_after=0):
        recent_rewards = []  # Track recent rewards for summary
        for episode in range(episodes):
            state = self.game.reset()
            done = False
            total_reward = 0
            visualize = (episode % visualize_interval == 0 and episode >= start_visualize_after)  

            logger.info(f"Starting episode {episode}{' (visualizing)' if visualize else ''}")
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.game.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                if visualize:
                    self.game.render()
                    await asyncio.sleep(1.0 / self.game.fps)  # Control frame rate only when rendering

            # Decay exploration rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            recent_rewards.append(total_reward)

            # Log episode metrics
            avg_q_value = torch.mean(self.q_table).item()
            logger.info(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.3f}, Avg Q-value: {avg_q_value:.3f}")

            # Log summary every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward = sum(recent_rewards[-100:]) / min(len(recent_rewards), 100)
                logger.info(f"Summary (Episodes {episode-99 if episode >= 99 else 0}-{episode}): Avg Reward: {avg_reward:.3f}, Epsilon: {self.epsilon:.3f}, Avg Q-value: {avg_q_value:.3f}")
                recent_rewards = []  # Reset for next summary

            # Save model periodically
            # if (episode + 1) % save_interval == 0:
            #     self.save_model()

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
            await asyncio.sleep(1.0 / self.game.fps)  # Control frame rate

        self.game.close()