import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import cv2
import matplotlib.pyplot as plt

# Image Preprocessing
def preprocess(img):
    # Crop bottom control bar, take 84x84 region, convert to grayscale
    img = img[:84, 6:90]  # CarRacing-v3 specific cropping
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0
    return img

# Environment Wrapper for Frame Stacking
class ImageEnv(gym.Wrapper):
    def __init__(self, env, stack_frames=4):
        super(ImageEnv, self).__init__(env)
        self.stack_frames = stack_frames
        self.frame_stack = deque(maxlen=stack_frames)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(stack_frames, 84, 84), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = preprocess(obs)
        for _ in range(self.stack_frames):
            self.frame_stack.append(obs)
        return np.stack(self.frame_stack, axis=0), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = preprocess(obs)
        self.frame_stack.append(obs)
        done = terminated or truncated
        return np.stack(self.frame_stack, axis=0), reward, done, info

# Dueling DQN Network (CNN-based)
class DuelingDQN(nn.Module):
    def __init__(self, state_shape, action_dim):
        super(DuelingDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        # Conv output: (84-8)/4+1=20, (20-4)/2+1=9, (9-3)/1+1=7 → 64*7*7
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.value = nn.Linear(512, 1)
        self.advantage = nn.Linear(512, action_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc(x))
        value = self.value(x)
        advantage = self.advantage(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

# Agent Class
class DQNAgent:
    def __init__(self, state_shape, action_dim, device):
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.device = device
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # Faster decay for quicker learning
        self.batch_size = 32
        self.memory = deque(maxlen=10000)
        
        self.model = DuelingDQN(state_shape, action_dim).to(device)
        self.target_model = DuelingDQN(state_shape, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# Training Loop with Periodic Rendering
env = gym.make("CarRacing-v3", render_mode=None, continuous=False)  # Default: no rendering
env = ImageEnv(env, stack_frames=4)
state_shape = env.observation_space.shape
action_dim = env.action_space.n
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DQNAgent(state_shape, action_dim, device)

episodes = 500
rewards = []
recent_rewards = deque(maxlen=10)  # Track last 10 episodes
update_target_freq = 10
max_steps = 1000  # Limit steps per episode
solved_reward = 900  # Threshold for solving
render_freq = 10  # Render every 10th episode

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(rewards)
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.set_title('Training Progress')

for episode in range(episodes):
    # Enable rendering for every 10th episode
    if episode % render_freq == 0:
        env.env.close()  # Close previous env if open
        env = gym.make("CarRacing-v3", render_mode="human", continuous=False)
        env = ImageEnv(env, stack_frames=4)
    
    state, _ = env.reset()
    total_reward = 0
    step = 0
    done = False
    while not done and step < max_steps:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward
        step += 1
    
    # Disable rendering after rendering episode
    if episode % render_freq == 0:
        env.env.close()
        env = gym.make("CarRacing-v3", render_mode=None, continuous=False)
        env = ImageEnv(env, stack_frames=4)
    
    agent.decay_epsilon()
    if episode % update_target_freq == 0:
        agent.update_target_model()
    rewards.append(total_reward)
    recent_rewards.append(total_reward)
    
    # Update live plot
    line.set_xdata(range(len(rewards)))
    line.set_ydata(rewards)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.01)
    
    # Check if solved
    avg_reward = np.mean(recent_rewards)
    print(f"Episode {episode}, Reward: {total_reward}, Steps: {step}, Epsilon: {agent.epsilon:.3f}, Avg Reward (last 10): {avg_reward:.2f}")
    if len(recent_rewards) == 10 and avg_reward > solved_reward:
        print(f"Environment solved at episode {episode}rial with average reward {avg_reward:.2f}")
        torch.save(agent.model.state_dict(), "dqn_car_racing_solved.pth")
        break

plt.ioff()
plt.show()
env.close()