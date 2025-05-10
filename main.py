import asyncio
import argparse
import torch
import logging
from environment import SoccerEnvironment
from agent import DQNAgent
from utils import setup_logging
import platform
import time
import os
import glob
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Q-learning soccer game")
    parser.add_argument("--mode", choices=["train", "continue", "play"], default="train", help="Mode: train, continue, or play (human or agent vs agent)")
    parser.add_argument("--render", action="store_true", help="Enable visual rendering")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players per team")
    parser.add_argument("--human-mode", action="store_true", help="Enable human control for Team A in play mode")
    return parser.parse_args()

def find_latest_checkpoint(team):
    # Find all checkpoint files in model/ directory
    pattern = f"model/dqn_{team}_ep*.pth"
    checkpoint_files = glob.glob(pattern)
    if not checkpoint_files:
        return None
    # Extract episode numbers and find the highest
    def get_episode_number(filename):
        match = re.search(r'ep(\d+)\.pth$', filename)
        return int(match.group(1)) if match else -1
    latest_file = max(checkpoint_files, key=get_episode_number)
    return latest_file

async def main():
    args = parse_args()
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    logging.info(f"Starting in {args.mode} mode with render={args.render}, num_players={args.num_players}, human_mode={args.human_mode}, device={device}")

    # Create model/ directory if it doesn't exist
    os.makedirs("model", exist_ok=True)

    # Initialize environment and agents
    env = SoccerEnvironment(render=args.render, num_players=args.num_players, human_mode=args.human_mode)
    state_dim = env.state_space
    action_dim = env.action_space
    team_a_agent = DQNAgent(state_dim, action_dim, lr=0.0005, gamma=0.99, epsilon=0.0, epsilon_decay=0.995, device=device)  # Epsilon=0 for play mode
    team_b_agent = DQNAgent(state_dim, action_dim, lr=0.0005, gamma=0.99, epsilon=0.0, epsilon_decay=0.995, device=device)  # Epsilon=0 for play mode

    # Load latest models for play mode or continue mode
    if args.mode in ["continue", "play"]:
        try:
            team_a_checkpoint = find_latest_checkpoint("team_a")
            team_b_checkpoint = find_latest_checkpoint("team_b")
            if team_a_checkpoint and team_b_checkpoint:
                team_a_agent.load(team_a_checkpoint)
                team_b_agent.load(team_b_checkpoint)
                logging.info(f"Loaded latest checkpoints: {team_a_checkpoint}, {team_b_checkpoint}")
            else:
                raise FileNotFoundError("No checkpoint files found")
        except FileNotFoundError:
            try:
                # Fall back to default models
                team_a_agent.load("model/dqn_team_a.pth")
                team_b_agent.load("model/dqn_team_b.pth")
                logging.info("Loaded default models: model/dqn_team_a.pth, model/dqn_team_b.pth")
            except FileNotFoundError:
                logging.error("No saved models or checkpoints found. Exiting.")
                return

    # Training or play loop
    num_episodes = 2000 if args.mode in ["train", "continue"] else 1
    save_interval = 100
    for episode in range(num_episodes):
        start_time = time.time()
        state = env.reset()
        done = False
        total_reward_a = 0
        total_reward_b = 0
        while not done:
            if args.mode == "play" and args.human_mode:
                # Human controls Team A, agent controls Team B
                action_a = env.get_human_action()
                if action_a is None:  # Default to agent's action if no human input
                    action_a = team_a_agent.act(state)
                action_b = team_b_agent.act(state)
            elif args.mode == "play":
                # Agent vs Agent: Team A and Team B use their trained models
                action_a = team_a_agent.act(state)
                action_b = team_b_agent.act(state)
            else:
                # Training or continue mode: Both teams use exploration
                action_a = team_a_agent.act(state)
                action_b = team_b_agent.act(state)

            next_state, reward_a, reward_b, done = env.step(action_a, action_b)
            if args.mode in ["train", "continue"]:
                # Store experiences and train only in train/continue modes
                team_a_agent.remember(state, action_a, reward_a, next_state, done)
                team_b_agent.remember(state, action_b, reward_b, next_state, done)
                team_a_agent.replay()
                team_b_agent.replay()
            state = next_state
            total_reward_a += reward_a
            total_reward_b += reward_b
            if args.render:
                env.render()
                await asyncio.sleep(1.0 / 60)
        episode_duration = time.time() - start_time
        logging.info(f"Episode {episode + 1}/{num_episodes}, Team A Reward: {total_reward_a:.2f}, Team B Reward: {total_reward_b:.2f}, "
                     f"Score: {env.score_a}-{env.score_b}, Epsilon: {team_a_agent.epsilon:.4f}, Duration: {episode_duration:.2f}s")
        if args.mode in ["train", "continue"]:
            team_a_agent.decay_epsilon()
            team_b_agent.decay_epsilon()
            if (episode + 1) % save_interval == 0:
                team_a_agent.save(f"model/dqn_team_a_ep{episode + 1}.pth")
                team_b_agent.save(f"model/dqn_team_b_ep{episode + 1}.pth")
                logging.info(f"Saved models at episode {episode + 1}")

    # Save final models
    if args.mode in ["train", "continue"]:
        team_a_agent.save("model/dqn_team_a.pth")
        team_b_agent.save("model/dqn_team_b.pth")
        logging.info("Saved final models")
    env.close()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())