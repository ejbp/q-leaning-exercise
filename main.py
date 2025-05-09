import asyncio
import argparse
import torch
import logging
from environment import SoccerEnvironment
from agent import DQNAgent
from utils import setup_logging
import platform

def parse_args():
    parser = argparse.ArgumentParser(description="Q-learning soccer game")
    parser.add_argument("--mode", choices=["train", "continue"], default="train", help="Training mode")
    parser.add_argument("--render", action="store_true", help="Enable visual rendering for debugging")
    return parser.parse_args()

async def main():
    args = parse_args()
    setup_logging()
    logging.info(f"Starting in {args.mode} mode with render={args.render}")

    # Initialize environment and agent
    env = SoccerEnvironment(render=args.render)
    state_dim = env.state_space  # Player, opponent, ball, goal positions
    action_dim = env.action_space  # Move up/down/left/right/kick
    agent = DQNAgent(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, device="cpu")

    # Load model if continuing
    if args.mode == "continue":
        try:
            agent.load("dqn_model.pth")
            logging.info("Loaded saved model")
        except FileNotFoundError:
            logging.warning("No saved model found, starting fresh")

    # Training loop
    num_episodes = 2000  # Increased for more complex environment
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            if args.render:
                env.render()
                await asyncio.sleep(1.0 / 60)  # Cap at 60 FPS
        logging.info(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")
        agent.decay_epsilon()

    # Save model
    agent.save("dqn_model.pth")
    env.close()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())