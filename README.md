Soccer Q-Learning Game
This project implements a 2D soccer game using Pygame, where two teams compete to score goals. The game supports both AI training via Deep Q-Learning (DQN) and interactive play modes, including human vs. AI and AI vs. AI. The AI agents are trained to pursue the ball, protect their goal, and coordinate strategically, with a sophisticated reward system to encourage intelligent behavior.
Features

AI Training: Train DQN agents for both teams using prioritized experience replay and shaped rewards.
Play Modes:
Human vs. AI: Control Team A (one player at a time) against a trained AI Team B.
AI vs. AI: Watch two trained AI teams compete.


Enhanced AI Behavior: Rewards incentivize ball pursuit, goal protection, optimal spacing, and teamwork.
Rendering: Visualize the game with Pygame, including field, players, ball, goals, and HUD (score/time).
Checkpointing: Save and load model checkpoints for continued training or play.

Requirements

Python 3.8+
Pygame (pip install pygame)
PyTorch (pip install torch)
NumPy (pip install numpy)

Installation

Clone the repository:git clone <repository-url>
cd soccer-q-learning


Create a virtual environment (optional but recommended):python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt

Alternatively, install manually:pip install pygame torch numpy



Usage
The game can be run in three modes: train, continue, or play. Use the command-line arguments to configure the mode, rendering, number of players, and human control.
Command-Line Arguments

--mode: Choose the mode (train, continue, play). Default: train.
--render: Enable visual rendering with Pygame. Default: disabled.
--num-players: Number of players per team (1-5 recommended). Default: 2.
--human-mode: Enable human control for Team A in play mode. Default: disabled.

Examples

Train AI from scratch:
python main.py --mode train --render --num-players 2


Trains DQN agents for 2000 episodes.
Saves checkpoints every 100 episodes in model/ (e.g., dqn_team_a_ep100.pth).
Final models saved as model/dqn_team_a.pth and model/dqn_team_b.pth.


Continue training from latest checkpoint:
python main.py --mode continue --render --num-players 2


Loads the latest checkpoint or default models from model/.
Continues training for 2000 episodes.


Play: Human vs. AI:
python main.py --mode play --render --num-players 2 --human-mode


Human controls Team A (one player at a time) using keyboard inputs.
Team B uses the latest trained model.
Controls:
Arrow keys: Move (Up, Down, Left, Right).
Space: Kick the ball.
Tab: Switch between players.
Quit: Close the Pygame window.




Play: AI vs. AI:
python main.py --mode play --render --num-players 2


Team A and Team B use their latest trained models.
No human input required; watch the AI teams compete.



Notes

Model Files: Ensure trained models exist in model/ for play or continue modes. Run train mode first if none exist.
Rendering: Use --render for visualization. Without it, the game runs in the background (useful for faster training).
Human Mode: Only one player is controlled at a time in --human-mode. Switching players with Tab allows strategic control.

File Structure

main.py: Entry point; handles mode selection, environment setup, and game loop.
environment.py: Defines the SoccerEnvironment class for game logic, physics, rendering, and reward shaping.
agent.py: Implements the DQNAgent class with DQN model, prioritized replay buffer, and training logic.
game.py: Alternative grid-based soccer environment (not used in the main game but included for reference).
utils.py: Utility functions, including logging setup.
model/: Directory for saving model checkpoints and final models.
training.log: Log file for training and game events.

Reward System
The AI is trained with a shaped reward system to encourage strategic play:

Ball Proximity: Higher rewards for players closer to the ball (weight: 1.0).
Goal Protection: Rewards for staying near own goal when the ball is in own half (weight: 2.0).
Teamwork Spacing: Rewards for maintaining optimal distance between teammates (50-200 pixels) and penalties for overcrowding.
Ball Progress: Rewards for moving the ball toward the opponent's goal (weight: 10.0).
Passing: Rewards for kicks near the goal or to better-positioned teammates.
Running: Rewards for avoiding opponents.
Goals: Large rewards (+100) for scoring, penalties (-100) for conceding.

Limitations

Human control is limited to one player at a time in --human-mode. Controlling all players simultaneously would require additional input handling.
Training can be computationally intensive; use a GPU with CUDA for faster training if available.
The grid-based game.py is not integrated with the main game but can be adapted for alternative environments.

Future Improvements

Allow simultaneous control of all players in human mode.
Add difficulty levels for AI opponents in play mode.
Implement more advanced AI strategies, such as intercepting passes or dynamic role assignment (e.g., striker, defender).
Support multiplayer human teams.

Contributing
Contributions are welcome! Please submit issues or pull requests to the repository. Ensure code follows PEP 8 style guidelines and includes tests where applicable.
License
This project is licensed under the MIT License. See the LICENSE file for details.
