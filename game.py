import pygame
import random
import asyncio

class GridGame:
    def __init__(self, size=5, cell_size=50, max_steps=300):
        self.max_steps = max_steps
        self.size = size
        self.cell_size = cell_size
        self.goal_pos = self._get_random_start()
        self.current_pos = self._get_random_start(goal_pos=self.goal_pos)
        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.action_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.total_steps = 0

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((size * cell_size, size * cell_size))
        pygame.display.set_caption("Grid Game")
        self.clock = pygame.time.Clock()
        self.fps = 10  # Frames per second for rendering

    def _get_random_start(self, goal_pos=None):
        # Generate all possible positions except the goal
        possible_starts = [(r, c) for r in range(self.size) for c in range(self.size) if goal_pos == None or (r, c) != goal_pos]
        return random.choice(possible_starts)

    def reset(self):
        self.goal_pos = self._get_random_start()
        self.current_pos = self._get_random_start(goal_pos=self.goal_pos)
        self.total_steps = 0
        return self.goal_pos, self.current_pos

    def _get_normalized_distance(self):
        # Calculate Manhattan distance
        manhattan_dist = abs(self.current_pos[0] - self.goal_pos[0]) + abs(self.current_pos[1] - self.goal_pos[1])
        # Maximum possible Manhattan distance is (size-1) + (size-1)
        max_dist = 2 * (self.size - 1)
        # Normalize distance to [0, 1], where 1 is max distance
        return manhattan_dist / max_dist if max_dist > 0 else 0

    def step(self, action):
        self.total_steps += 1

        # Get movement delta
        delta = self.action_map[action]
        new_pos = (self.current_pos[0] + delta[0], self.current_pos[1] + delta[1])

        # Check if new position is within grid bounds
        reward = 0
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            self.current_pos = new_pos
        else:
            reward = -1  # Penalty for hitting the wall

        # Calculate normalized distance penalty
        distance_penalty = -1 * self._get_normalized_distance()

        if self.max_steps > 0 and self.total_steps >= self.max_steps:
            done = True
            reward += -1000
        else:
            reward += 1000 if self.current_pos == self.goal_pos else -1 + distance_penalty
            done = self.current_pos == self.goal_pos

        return (self.goal_pos, self.current_pos), reward, done

    def render(self):
        # Handle Pygame events (e.g., close window)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # Clear screen
        self.screen.fill((255, 255, 255))  # White background

        # Draw grid
        for row in range(self.size):
            for col in range(self.size):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)  # Black border
                if (row, col) == self.current_pos:
                    pygame.draw.rect(self.screen, (0, 0, 255), rect.inflate(-10, -10))  # Blue agent
                elif (row, col) == self.goal_pos:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect.inflate(-10, -10))  # Green goal

        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()