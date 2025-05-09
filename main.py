import asyncio
import argparse
import torch
import logging
from environment import SoccerEnvironment
from agent import DQNAgent
from utils import setup_logging
import platform
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Q-learning soccer game")
    parser.add_argument("--mode", choices=["train", "continue", "play"], default="train", help="Mode: train, continue, or play (human mode)")
    parser.add_argument("--render", action="store_true", help="Enable visual rendering")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players per team")
    parser.add_argument("--human-mode", action="store_true", help="Enable human control for one player")
    return parser.parse_args()

async def main():
    args = parse_args()
    setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Starting in {args.mode} mode with render={args.render}, num_players={args.num_players}, human_mode={args.human_mode}, device={device}")

    # Initialize environment and agents
    env = SoccerEnvironment(render=args.render, num_players=args.num_players, human_mode=args.human_mode)
    state_dim = env.state_space
    action_dim = env.action_space
    team_a_agent = DQNAgent(state_dim, action_dim, lr=0.0005, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, device=device)
    team_b_agent = DQNAgent(state_dim, action_dim, lr=0.0005, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, device=device)

    # Load models if continuing
    if args.mode == "continue":
        try:
            team_a_agent.load("dqn_team_a.pth")
            team_b_agent.load("dqn_team_b.pth")
            logging.info("Loaded saved models for Team A and Team B")
        except FileNotFoundError:
            logging.warning("No saved models found, starting fresh")

    # Training or play loop
    num_episodes = 2000 if args.mode in ["train", "continue"] else 1
    save_interval = 10
    for episode in range(num_episodes):
        start_time = time.time()
        state = env.reset()
        done = False
        total_reward_a = 0
        total_reward_b = 0
        while not done:
            if args.human_mode and args.mode == "play":
                action_a = env.get_human_action()
                action_b = team_b_agent.act(state)
            else:
                action_a = team_a_agent.act(state)
                action_b = team_b_agent.act(state)
            next_state, reward_a, reward_b, done = env.step(action_a, action_b)
            if args.mode in ["train", "continue"]:
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
                team_a_agent.save(f"dqn_team_a_ep{episode + 1}.pth")
                team_b_agent.save(f"dqn_team_b_ep{episode + 1}.pth")
                logging.info(f"Saved models at episode {episode + 1}")

    # Save final models
    if args.mode in ["train", "continue"]:
        team_a_agent.save("dqn_team_a.pth")
        team_b_agent.save("dqn_team_b.pth")
        logging.info("Saved final models")
    env.close()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())