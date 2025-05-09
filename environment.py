import pygame
import numpy as np

class SoccerEnvironment:
    def __init__(self, render=False):
        self.field_width, self.field_height = 800, 600
        self.goal_width, self.goal_height = 50, 200
        self.player_size = 20
        self.ball_size = 10
        self.render_enabled = render
        self.screen = pygame.display.set_mode((self.field_width, self.field_height)) if render else None
        self.clock = pygame.time.Clock() if render else None

        # Game state
        self.player_pos = [self.field_width // 4, self.field_height // 2]  # Start near left side
        self.ball_pos = [self.field_width // 2, self.field_height // 2]    # Start at center
        self.left_goal = (0, (self.field_height - self.goal_height) // 2)
        self.right_goal = (self.field_width - self.goal_width, (self.field_height - self.goal_height) // 2)

        # Action and state spaces
        self.action_space = 4  # Up, down, left, right
        self.state_space = 6   # Player (x, y), Ball (x, y), Goal (x, y) (simplified to one goal for now)
        self.reset()

    def reset(self):
        self.player_pos = [self.field_width // 4, self.field_height // 2]
        self.ball_pos = [self.field_width // 2, self.field_height // 2]
        return self.get_state()

    def get_state(self):
        # Simplified state: player and ball positions, targeting right goal
        return np.array([
            self.player_pos[0], self.player_pos[1],
            self.ball_pos[0], self.ball_pos[1],
            self.right_goal[0] + self.goal_width // 2, self.right_goal[1] + self.goal_height // 2
        ])

    def step(self, action):
        # Move player
        speed = 5
        if action == 0:  # Up
            self.player_pos[1] = max(self.player_size, self.player_pos[1] - speed)
        elif action == 1:  # Down
            self.player_pos[1] = min(self.field_height - self.player_size, self.player_pos[1] + speed)
        elif action == 2:  # Left
            self.player_pos[0] = max(self.player_size, self.player_pos[0] - speed)
        elif action == 3:  # Right
            self.player_pos[0] = min(self.field_width - self.player_size, self.player_pos[0] + speed)

        # Check if player is near ball to "kick" it
        reward = -0.1  # Small negative reward per step
        done = False
        player_rect = pygame.Rect(self.player_pos[0] - self.player_size, self.player_pos[1] - self.player_size, self.player_size * 2, self.player_size * 2)
        ball_rect = pygame.Rect(self.ball_pos[0] - self.ball_size, self.ball_pos[1] - self.ball_size, self.ball_size * 2, self.ball_size * 2)
        if player_rect.colliderect(ball_rect):
            # Move ball towards right goal
            dx = self.right_goal[0] + self.goal_width // 2 - self.ball_pos[0]
            dy = self.right_goal[1] + self.goal_height // 2 - self.ball_pos[1]
            dist = (dx**2 + dy**2)**0.5
            if dist > 0:
                self.ball_pos[0] += (dx / dist) * 10
                self.ball_pos[1] += (dy / dist) * 10

        # Check if ball is in goal
        ball_rect = pygame.Rect(self.ball_pos[0] - self.ball_size, self.ball_pos[1] - self.ball_size, self.ball_size * 2, self.ball_size * 2)
        right_goal_rect = pygame.Rect(self.right_goal[0], self.right_goal[1], self.goal_width, self.goal_height)
        if ball_rect.colliderect(right_goal_rect):
            reward = 100
            done = True

        # Check if ball is out of bounds
        if (self.ball_pos[0] < 0 or self.ball_pos[0] > self.field_width or
                self.ball_pos[1] < 0 or self.ball_pos[1] > self.field_height):
            reward = -100
            done = True

        next_state = self.get_state()
        return next_state, reward, done

    def draw_field(self, screen):
        # Alternating green stripes
        stripe_width = 50
        for x in range(0, self.field_width, stripe_width):
            color = (0, 120, 0) if (x // stripe_width) % 2 == 0 else (0, 100, 0)
            pygame.draw.rect(screen, color, (x, 0, stripe_width, self.field_height))

    def draw_field_lines(self, screen):
        white = (255, 255, 255)
        line_width = 4
        # Halfway line
        pygame.draw.line(screen, white, (self.field_width // 2, 0), (self.field_width // 2, self.field_height), line_width)
        # Center circle
        pygame.draw.circle(screen, white, (self.field_width // 2, self.field_height // 2), 50, line_width)
        # Penalty areas
        penalty_width, penalty_height = 150, 300
        pygame.draw.rect(screen, white, (0, (self.field_height - penalty_height) // 2, penalty_width, penalty_height), line_width)
        pygame.draw.rect(screen, white, (self.field_width - penalty_width, (self.field_height - penalty_height) // 2, penalty_width, penalty_height), line_width)
        # Goal areas
        goal_area_width, goal_area_height = 50, 200
        pygame.draw.rect(screen, white, (0, (self.field_height - goal_area_height) // 2, goal_area_width, goal_area_height), line_width)
        pygame.draw.rect(screen, white, (self.field_width - goal_area_width, (self.field_height - goal_area_height) // 2, goal_area_width, goal_area_height), line_width)

    def draw_goals(self, screen):
        # Draw goals (white rectangles)
        pygame.draw.rect(screen, (255, 255, 255), (*self.left_goal, self.goal_width, self.goal_height))
        pygame.draw.rect(screen, (255, 255, 255), (*self.right_goal, self.goal_width, self.goal_height))
        # Debug: Red dots at goal centers
        pygame.draw.circle(screen, (255, 0, 0), (self.left_goal[0] + self.goal_width // 2, self.left_goal[1] + self.goal_height // 2), 5)
        pygame.draw.circle(screen, (255, 0, 0), (self.right_goal[0] + self.goal_width // 2, self.right_goal[1] + self.goal_height // 2), 5)

    def draw_players(self, screen):
        # Draw player as a blue circle
        pygame.draw.circle(screen, (0, 0, 255), (int(self.player_pos[0]), int(self.player_pos[1])), self.player_size)

    def draw_ball(self, screen):
        # Draw ball as a red circle
        pygame.draw.circle(screen, (255, 0, 0), (int(self.ball_pos[0]), int(self.ball_pos[1])), self.ball_size)

    def render(self):
        if not self.render_enabled:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
        self.draw_field(self.screen)
        self.draw_field_lines(self.screen)
        self.draw_players(self.screen)
        self.draw_ball(self.screen)
        self.draw_goals(self.screen)  # Goals drawn last to appear on top
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.render_enabled:
            pygame.quit()