import pygame
import random
import asyncio

class GridGame:
    def __init__(self, size=5, cell_size=None, max_steps=None):
        self.size = size
        self.cell_size = cell_size if cell_size is not None else max(10, 500 // size)  # Dynamic cell size
        self.max_steps = max_steps if max_steps is not None else 4 * size  # Dynamic max steps
        self.goal_pos = self._get_random_start()
        self.current_pos = self._get_random_start(goal_pos=self.goal_pos)
        self.actions = [0, 1, 2, 3]  # Up, Right, Down, Left
        self.action_map = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        self.total_steps = 0

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((size * self.cell_size, size * self.cell_size))
        pygame.display.set_caption("Grid Game")
        self.clock = pygame.time.Clock()
        self.fps = 10  # Frames per second for rendering

    def _get_random_start(self, goal_pos=None):
        possible_starts = [(r, c) for r in range(self.size) for c in range(self.size) if goal_pos is None or (r, c) != goal_pos]
        return random.choice(possible_starts)

    def reset(self):
        self.goal_pos = self._get_random_start()
        self.current_pos = self._get_random_start(goal_pos=self.goal_pos)
        self.total_steps = 0
        return self.goal_pos, self.current_pos

    def _get_normalized_distance(self):
        manhattan_dist = abs(self.current_pos[0] - self.goal_pos[0]) + abs(self.current_pos[1] - self.goal_pos[1])
        max_dist = 2 * (self.size - 1)
        return 10 * (manhattan_dist / max_dist) if max_dist > 0 else 0  # Scaled by 10

    def step(self, action):
        self.total_steps += 1

        delta = self.action_map[action]
        new_pos = (self.current_pos[0] + delta[0], self.current_pos[1] + delta[1])

        reward = 0
        if 0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size:
            self.current_pos = new_pos
        else:
            reward = -10  # Stronger wall penalty

        distance_penalty = -1 * self._get_normalized_distance()

        if self.max_steps > 0 and self.total_steps >= self.max_steps:
            done = True
            reward += -1000
        else:
            reward += 1000 if self.current_pos == self.goal_pos else -1 + distance_penalty
            done = self.current_pos == self.goal_pos

        return (self.goal_pos, self.current_pos), reward, done

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        self.screen.fill((255, 255, 255))

        for row in range(self.size):
            for col in range(self.size):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
                if (row, col) == self.current_pos:
                    pygame.draw.rect(self.screen, (0, 0, 255), rect.inflate(-10, -10))
                elif (row, col) == self.goal_pos:
                    pygame.draw.rect(self.screen, (0, 255, 0), rect.inflate(-10, -10))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()