import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import heapq
import logging

torch._dynamo.config.suppress_errors = True

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.memory = []
        self.priorities = []
        self.pos = 0

    def push(self, transition, priority):
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
            self.priorities.append(priority)
        else:
            self.memory[self.pos] = transition
            self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.0005, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.model = DQN(state_dim, action_dim).to(self.device)
        self.target_model = DQN(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(capacity=10000)
        self.batch_size = 64
        self.tau = 0.005

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return np.argmax(q_values.cpu().numpy())

    def remember(self, state, action, reward, next_state, done):
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        q_values = self.model(state_tensor)
        next_q_values = self.target_model(next_state_tensor).max(0)[0]
        td_error = abs(reward + (1 - done) * self.gamma * next_q_values - q_values[action]).item() + 1e-5
        self.memory.push((state, action, reward, next_state, done), td_error)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_model(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = (weights * nn.MSELoss(reduction='none')(current_q, target_q.detach())).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        td_errors = torch.abs(current_q - target_q).detach().cpu().numpy() + 1e-5
        self.memory.update_priorities(indices, td_errors)

        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, path)
        logging.info(f"Saved model and epsilon to {path}")

    def load(self, path):
        logging.info(f"Loading model and epsilon from {path} to device {self.device}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.target_model.load_state_dict(self.model.state_dict())
        self.epsilon = checkpoint.get('epsilon', self.epsilon)  # Load epsilon, fallback to current