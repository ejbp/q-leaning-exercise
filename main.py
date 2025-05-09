import argparse
import asyncio
import platform
from game import GridGame
from agent import DQNAgent
from utils import setup_logging, setup_device

logger = setup_logging()

async def main():
    parser = argparse.ArgumentParser(description="Grid Game DQN with CUDA and Pygame")
    parser.add_argument("--mode", choices=["train", "continue", "play"], default="train", help="Mode: train (new model), continue (load and train), play (inference)")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of training episodes")
    parser.add_argument("--save-interval", type=int, default=300, help="Save model every N episodes")
    parser.add_argument("--visualize-interval", type=int, default=300, help="Visualize game every N episodes during training")
    parser.add_argument("--model-path", type=str, default="dqn_model.pt", help="Path to save/load DQN model")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DQN training")
    parser.add_argument("--replay-size", type=int, default=10000, help="Size of the replay buffer")
    parser.add_argument("--target-update", type=int, default=1000, help="Steps between target network updates")
    args = parser.parse_args()

    # Initialize game and agent
    game = GridGame(size=20, cell_size=50, max_steps=80)
    agent = DQNAgent(
        game,
        model_path=args.model_path,
        device=setup_device(),
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        target_update=args.target_update
    )

    try:
        if args.mode == "train":
            logger.info("Starting training from scratch")
            await agent.train(
                episodes=args.episodes,
                save_interval=args.save_interval,
                visualize_interval=args.visualize_interval,
                start_visualize_after=100
            )
        elif args.mode == "continue":
            import os
            if not os.path.exists(args.model_path):
                logger.warning(f"No model found at {args.model_path}. Starting fresh training.")
            logger.info("Continuing training from saved model")
            await agent.train(
                episodes=args.episodes,
                save_interval=args.save_interval,
                visualize_interval=args.visualize_interval
            )
        elif args.mode == "play":
            import os
            if not os.path.exists(args.model_path):
                logger.error(f"No model found at {args.model_path}. Cannot play.")
                return
            agent.epsilon = 0.0  # Disable exploration for optimal play
            logger.info("Starting play mode with saved model")
            await agent.play()
    except SystemExit:
        logger.info("Window closed. Exiting.")
        raise
    finally:
        game.close()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())