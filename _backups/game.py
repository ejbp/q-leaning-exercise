import pygame
import random
import asyncio
import numpy as np

class SoccerGame:
    def __init__(self, rows=20, cols=40, cell_size=None, max_steps=1000, num_players=3, enable_graphics=False):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size if cell_size is not None else max(10, 800 // cols)  # Fit 800x400
        self.max_steps = max_steps
        self.num_players = num_players
        self.actions = [0, 1, 2, 3, 4]  # Up, Right, Down, Left, Kick
        self.action_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # Movement deltas
        self.kick_directions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}  # Kick directions
        self.total_steps = 0
        self.score_a = 0
        self.score_b = 0
        self.enable_graphics = enable_graphics

        # Initialize positions
        self.team_a_positions = []
        self.team_b_positions = []
        self.ball_pos = (rows // 2, cols // 2)
        self.goal_a = [(0, c) for c in range(cols // 4, cols // 4 + 4)]  # Team B's goal (top edge)
        self.goal_b = [(rows-1, c) for c in range(3 * cols // 4 - 4, 3 * cols // 4)]  # Team A's goal (bottom edge)
        self.reset()

        # Human control state
        self.selected_player = 0  # Index of human-controlled player
        self.human_action = None

        # Initialize Pygame only if graphics are enabled
        if self.enable_graphics:
            pygame.init()
            self.screen = pygame.display.set_mode((cols * self.cell_size, rows * self.cell_size))
            pygame.display.set_caption("Soccer Game")
            self.clock = pygame.time.Clock()
            self.fps = 10
            self.font = pygame.font.SysFont("arial", 20)

    def _get_random_start(self, team="A"):
        # Random positions on team's half
        half_cols = self.cols // 2
        if team == "A":
            possible_starts = [(r, c) for r in range(self.rows) for c in range(half_cols)]
        else:
            possible_starts = [(r, c) for r in range(self.rows) for c in range(half_cols, self.cols)]
        # Exclude goal areas
        possible_starts = [p for p in possible_starts if p not in self.goal_a and p not in self.goal_b]
        return possible_starts

    def reset(self):
        self.total_steps = 0
        self.score_a = 0
        self.score_b = 0
        self.team_a_positions = [random.choice(self._get_random_start("A")) for _ in range(self.num_players)]
        self.team_b_positions = [random.choice(self._get_random_start("B")) for _ in range(self.num_players)]
        self.ball_pos = (self.rows // 2, self.cols // 2)
        self.selected_player = 0
        self.human_action = None
        return self.get_state()

    def get_state(self):
        return (self.team_a_positions, self.team_b_positions, self.ball_pos, self.goal_a, self.goal_b)

    def _is_adjacent(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) <= 1

    def _get_ball_progress(self):
        # Normalized ball position toward Team B's goal (x=0 is max progress for Team A)
        return 1 - (self.ball_pos[1] / (self.cols - 1))  # Ranges from 0 (Team A's goal) to 1 (Team B's goal)

    def get_human_actions(self):
        # Return last human action for selected player, random for others
        actions = [random.choice(self.actions) for _ in range(self.num_players)]
        if self.human_action is not None:
            actions[self.selected_player] = self.human_action
        return actions

    def step(self, team_a_actions, team_b_actions):
        self.total_steps += 1
        reward = 0
        goal_scored = False

        # Move Team A players
        for i, action in enumerate(team_a_actions):
            if action < 4:  # Movement
                delta = self.action_map[action]
                new_pos = (self.team_a_positions[i][0] + delta[0], self.team_a_positions[i][1] + delta[1])
                if 0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols and new_pos not in self.goal_a + self.goal_b:
                    self.team_a_positions[i] = new_pos
            elif action == 4 and self._is_adjacent(self.team_a_positions[i], self.ball_pos):  # Kick
                kick_dir = self.kick_directions[random.choice([0, 1, 2, 3])]  # Random kick direction
                new_ball_pos = (self.ball_pos[0] + kick_dir[0], self.ball_pos[1] + kick_dir[1])
                if 0 <= new_ball_pos[0] < self.rows and 0 <= new_ball_pos[1] < self.cols:
                    self.ball_pos = new_ball_pos

        # Move Team B players
        for i, action in enumerate(team_b_actions):
            if action < 4:
                delta = self.action_map[action]
                new_pos = (self.team_b_positions[i][0] + delta[0], self.team_b_positions[i][1] + delta[1])
                if 0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols and new_pos not in self.goal_a + self.goal_b:
                    self.team_b_positions[i] = new_pos
            elif action == 4 and self._is_adjacent(self.team_b_positions[i], self.ball_pos):
                kick_dir = self.kick_directions[random.choice([0, 1, 2, 3])]
                new_ball_pos = (self.ball_pos[0] + kick_dir[0], self.ball_pos[1] + kick_dir[1])
                if 0 <= new_ball_pos[0] < self.rows and 0 <= new_ball_pos[1] < self.cols:
                    self.ball_pos = new_ball_pos

        # Check for goals
        if self.ball_pos in self.goal_a:  # Team B scores
            self.score_b += 1
            reward = -100
            goal_scored = True
            self.team_a_positions = [random.choice(self._get_random_start("A")) for _ in range(self.num_players)]
            self.team_b_positions = [random.choice(self._get_random_start("B")) for _ in range(self.num_players)]
            self.ball_pos = (self.rows // 2, self.cols // 2)
        elif self.ball_pos in self.goal_b:  # Team A scores
            self.score_a += 1
            reward = 100
            goal_scored = True
            self.team_a_positions = [random.choice(self._get_random_start("A")) for _ in range(self.num_players)]
            self.team_b_positions = [random.choice(self._get_random_start("B")) for _ in range(self.num_players)]
            self.ball_pos = (self.rows // 2, self.cols // 2)

        # Shaped reward: progress toward Team B's goal
        if not goal_scored:
            reward += 10 * self._get_ball_progress()  # Encourage moving ball toward x=0

        # Penalty for time step
        reward -= 0.1

        # Check if match is over
        done = self.total_steps >= self.max_steps
        info = {"score_a": self.score_a, "score_b": self.score_b}

        return self.get_state(), reward, done, info

    def render(self):
        if not self.enable_graphics:
            return  # Skip rendering if graphics are disabled

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.human_action = 0
                elif event.key == pygame.K_RIGHT:
                    self.human_action = 1
                elif event.key == pygame.K_DOWN:
                    self.human_action = 2
                elif event.key == pygame.K_LEFT:
                    self.human_action = 3
                elif event.key == pygame.K_SPACE:
                    self.human_action = 4  # Kick
                elif event.key == pygame.K_TAB:
                    self.selected_player = (self.selected_player + 1) % self.num_players  # Switch player

        self.screen.fill((0, 100, 0))  # Green field

        # Draw grid
        for row in range(self.rows):
            for col in range(self.cols):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 1)

        # Draw goals
        for pos in self.goal_a:
            rect = pygame.Rect(pos[1] * self.cell_size, pos[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 255, 0), rect)  # Yellow goal
        for pos in self.goal_b:
            rect = pygame.Rect(pos[1] * self.cell_size, pos[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 255, 0), rect)

        # Draw players
        for i, pos in enumerate(self.team_a_positions):
            rect = pygame.Rect(pos[1] * self.cell_size, pos[0] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (0, 0, 255), rect.inflate(-5, -5))  # Blue for Team A
        for i, pos in enumerate(self.team_b_positions):
            rect = pygame.Rect(pos[1] * self.cell_size, pos[0] * self.cell_size, self.cell_size, self.cell_size)
            color = (255, 0, 0) if i != self.selected_player else (255, 165, 0)  # Red for Team B, orange for selected
            pygame.draw.rect(self.screen, color, rect.inflate(-5, -5))

        # Draw ball
        rect = pygame.Rect(self.ball_pos[1] * self.cell_size, self.ball_pos[0] * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.circle(self.screen, (255, 255, 255), rect.center, self.cell_size // 4)

        # Draw score and time
        time_left = max(0, self.max_steps - self.total_steps)
        score_text = self.font.render(f"Team A: {self.score_a}  Team B: {self.score_b}", True, (255, 255, 255))
        time_text = self.font.render(f"Time: {time_left}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (self.cols * self.cell_size - 100, 10))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if self.enable_graphics:
            pygame.quit()