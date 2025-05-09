import pygame
import numpy as np
import random

class SoccerEnvironment:
    def __init__(self, render=False, num_players=2, human_mode=False):
        self.field_width, self.field_height = 800, 600
        self.goal_width, self.goal_height = 50, 200
        self.player_size = 20
        self.ball_size = 10
        self.render_enabled = render
        self.num_players = num_players
        self.human_mode = human_mode
        self.max_steps = 1000
        self.current_step = 0
        self.score_a = 0
        self.score_b = 0
        self.selected_player = 0
        self.human_action = None
        self.screen = pygame.display.set_mode((self.field_width, self.field_height)) if render else None
        self.clock = pygame.time.Clock() if render else None
        self.font = pygame.font.SysFont("arial", 20) if render else None

        self.team_a_positions = []
        self.team_b_positions = []
        self.ball_pos = [self.field_width // 2, self.field_height // 2]
        self.ball_velocity = [0, 0]
        self.left_goal = (0, (self.field_height - self.goal_height) // 2)
        self.right_goal = (self.field_width - self.goal_width, (self.field_height - self.goal_height) // 2)

        self.action_space = 5
        self.state_space = 4 * num_players + 4 + 2
        self.reset()

    def reset(self):
        self.team_a_positions = [[self.field_width // 4, self.field_height // (self.num_players + 1) * (i + 1)] for i in range(self.num_players)]
        self.team_b_positions = [[3 * self.field_width // 4, self.field_height // (self.num_players + 1) * (i + 1)] for i in range(self.num_players)]
        self.ball_pos = [self.field_width // 2, self.field_height // 2]
        self.ball_velocity = [0, 0]
        self.current_step = 0
        self.selected_player = 0
        self.human_action = None
        return self.get_state()

    def get_state(self):
        state = []
        for pos in self.team_a_positions:
            state.extend([pos[0], pos[1]])
        for pos in self.team_b_positions:
            state.extend([pos[0], pos[1]])
        state.extend([self.ball_pos[0], self.ball_pos[1], self.ball_velocity[0], self.ball_velocity[1]])
        state.extend([self.right_goal[0] + self.goal_width // 2, self.right_goal[1] + self.goal_height // 2])
        return np.array(state)

    def get_human_action(self):
        if not self.human_mode:
            return None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.human_action = 0
                elif event.key == pygame.K_DOWN:
                    self.human_action = 1
                elif event.key == pygame.K_LEFT:
                    self.human_action = 2
                elif event.key == pygame.K_RIGHT:
                    self.human_action = 3
                elif event.key == pygame.K_SPACE:
                    self.human_action = 4
                elif event.key == pygame.K_TAB:
                    self.selected_player = (self.selected_player + 1) % self.num_players
        return self.human_action

    def step(self, action_a, action_b):
        self.current_step += 1
        team_a_actions = [action_a] * self.num_players
        team_b_actions = [action_b] * self.num_players
        if self.human_mode:
            human_action = self.get_human_action()
            if human_action is not None:
                team_a_actions[self.selected_player] = human_action

        # Move Team A players
        speed = 5
        for i, action in enumerate(team_a_actions):
            pos = self.team_a_positions[i]
            if action == 0:
                pos[1] = max(self.player_size, pos[1] - speed)
            elif action == 1:
                pos[1] = min(self.field_height - self.player_size, pos[1] + speed)
            elif action == 2:
                pos[0] = max(self.player_size, pos[0] - speed)
            elif action == 3:
                pos[0] = min(self.field_width - self.player_size, pos[0] + speed)
            elif action == 4:
                player_rect = pygame.Rect(pos[0] - self.player_size, pos[1] - self.player_size, self.player_size * 2, self.player_size * 2)
                ball_rect = pygame.Rect(self.ball_pos[0] - self.ball_size, self.ball_pos[1] - self.ball_size, self.ball_size * 2, self.ball_size * 2)
                if player_rect.colliderect(ball_rect):
                    dx = self.right_goal[0] + self.goal_width // 2 - self.ball_pos[0]
                    dy = self.right_goal[1] + self.goal_height // 2 - self.ball_pos[1]
                    dist = (dx**2 + dy**2)**0.5
                    if dist > 0:
                        kick_speed = 15
                        self.ball_velocity = [kick_speed * dx / dist, kick_speed * dy / dist]

        # Move Team B players
        for i, action in enumerate(team_b_actions):
            pos = self.team_b_positions[i]
            if action == 0:
                pos[1] = max(self.player_size, pos[1] - speed)
            elif action == 1:
                pos[1] = min(self.field_height - self.player_size, pos[1] + speed)
            elif action == 2:
                pos[0] = max(self.player_size, pos[0] - speed)
            elif action == 3:
                pos[0] = min(self.field_width - self.player_size, pos[0] + speed)
            elif action == 4:
                player_rect = pygame.Rect(pos[0] - self.player_size, pos[1] - self.player_size, self.player_size * 2, self.player_size * 2)
                ball_rect = pygame.Rect(self.ball_pos[0] - self.ball_size, self.ball_pos[1] - self.ball_size, self.ball_size * 2, self.ball_size * 2)
                if player_rect.colliderect(ball_rect):
                    dx = self.left_goal[0] + self.goal_width // 2 - self.ball_pos[0]
                    dy = self.left_goal[1] + self.goal_height // 2 - self.ball_pos[1]
                    dist = (dx**2 + dy**2)**0.5
                    if dist > 0:
                        kick_speed = 15
                        self.ball_velocity = [kick_speed * dx / dist, kick_speed * dy / dist]

        # Update ball physics
        self.ball_pos[0] += self.ball_velocity[0]
        self.ball_pos[1] += self.ball_velocity[1]
        friction = 0.98
        self.ball_velocity[0] *= friction
        self.ball_velocity[1] *= friction
        self.ball_pos[0] = max(self.ball_size, min(self.field_width - self.ball_size, self.ball_pos[0]))
        self.ball_pos[1] = max(self.ball_size, min(self.field_height - self.ball_size, self.ball_pos[1]))

        # Check collisions
        ball_rect = pygame.Rect(self.ball_pos[0] - self.ball_size, self.ball_pos[1] - self.ball_size, self.ball_size * 2, self.ball_size * 2)
        for i, pos in enumerate(self.team_a_positions):
            player_rect = pygame.Rect(pos[0] - self.player_size, pos[1] - self.player_size, self.player_size * 2, self.player_size * 2)
            for j, opp_pos in enumerate(self.team_b_positions):
                opponent_rect = pygame.Rect(opp_pos[0] - self.player_size, opp_pos[1] - self.player_size, self.player_size * 2, self.player_size * 2)
                if player_rect.colliderect(opponent_rect):
                    dx = pos[0] - opp_pos[0]
                    dy = pos[1] - opp_pos[1]
                    dist = (dx**2 + dy**2)**0.5
                    if dist > 0:
                        push = 2
                        pos[0] += push * dx / dist
                        pos[1] += push * dy / dist
                        opp_pos[0] -= push * dx / dist
                        opp_pos[1] -= push * dy / dist

        # Reward shaping
        reward_a = -0.1
        reward_b = -0.1
        min_dist_to_ball_a = min((pos[0] - self.ball_pos[0])**2 + (pos[1] - self.ball_pos[1])**2 for pos in self.team_a_positions)**0.5
        min_dist_to_ball_b = min((pos[0] - self.ball_pos[0])**2 + (pos[1] - self.ball_pos[1])**2 for pos in self.team_b_positions)**0.5
        ball_progress_a = 1 - self.ball_pos[0] / self.field_width
        ball_progress_b = self.ball_pos[0] / self.field_width
        reward_a += 0.5 * (1 - min_dist_to_ball_a / (self.field_width**2 + self.field_height**2)**0.5)
        reward_b += 0.5 * (1 - min_dist_to_ball_b / (self.field_width**2 + self.field_height**2)**0.5)
        reward_a += 10 * ball_progress_a
        reward_b += 10 * ball_progress_b

        # Passing reward (Team A)
        for i, pos in enumerate(self.team_a_positions):
            player_rect = pygame.Rect(pos[0] - self.player_size, pos[1] - self.player_size, self.player_size * 2, self.player_size * 2)
            if player_rect.colliderect(ball_rect):
                dist_to_goal = ((pos[0] - self.right_goal[0])**2 + (pos[1] - self.right_goal[1])**2)**0.5
                if dist_to_goal < 100:  # Near goal, prioritize shooting
                    reward_a += 5
                else:
                    for j, teammate_pos in enumerate(self.team_a_positions):
                        if i != j:
                            teammate_dist_to_goal = ((teammate_pos[0] - self.right_goal[0])**2 + (teammate_pos[1] - self.right_goal[1])**2)**0.5
                            if teammate_dist_to_goal < dist_to_goal:
                                reward_a += 2  # Reward passing to better-positioned teammate

        # Running reward (Team A)
        for pos in self.team_a_positions:
            min_dist_to_opponent = min((pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 for opp_pos in self.team_b_positions)**0.5
            if min_dist_to_opponent > 100:  # Open space
                reward_a += 1

        # Passing and running for Team B (mirrored)
        for i, pos in enumerate(self.team_b_positions):
            player_rect = pygame.Rect(pos[0] - self.player_size, pos[1] - self.player_size, self.player_size * 2, self.player_size * 2)
            if player_rect.colliderect(ball_rect):
                dist_to_goal = ((pos[0] - self.left_goal[0])**2 + (pos[1] - self.left_goal[1])**2)**0.5
                if dist_to_goal < 100:
                    reward_b += 5
                else:
                    for j, teammate_pos in enumerate(self.team_b_positions):
                        if i != j:
                            teammate_dist_to_goal = ((teammate_pos[0] - self.left_goal[0])**2 + (teammate_pos[1] - self.left_goal[1])**2)**0.5
                            if teammate_dist_to_goal < dist_to_goal:
                                reward_b += 2

        for pos in self.team_b_positions:
            min_dist_to_opponent = min((pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 for opp_pos in self.team_a_positions)**0.5
            if min_dist_to_opponent > 100:
                reward_b += 1

        # Goal and termination
        done = False
        right_goal_rect = pygame.Rect(self.right_goal[0], self.right_goal[1], self.goal_width, self.goal_height)
        left_goal_rect = pygame.Rect(self.left_goal[0], self.left_goal[1], self.goal_width, self.goal_height)

        if ball_rect.colliderect(right_goal_rect):
            self.score_a += 1
            reward_a = 100
            reward_b = -100
            done = True
        elif ball_rect.colliderect(left_goal_rect):
            self.score_b += 1
            reward_a = -100
            reward_b = 100
            done = True
        elif (self.ball_pos[0] <= self.ball_size or self.ball_pos[0] >= self.field_width - self.ball_size or
              self.ball_pos[1] <= self.ball_size or self.ball_pos[1] >= self.field_height - self.ball_size):
            reward_a = -50
            reward_b = -50
            done = True
        elif self.current_step >= self.max_steps:
            done = True

        next_state = self.get_state()
        return next_state, reward_a, reward_b, done

    def draw_field(self, screen):
        stripe_width = 50
        for x in range(0, self.field_width, stripe_width):
            color = (0, 120, 0) if (x // stripe_width) % 2 == 0 else (0, 100, 0)
            pygame.draw.rect(screen, color, (x, 0, stripe_width, self.field_height))

    def draw_field_lines(self, screen):
        white = (255, 255, 255)
        line_width = 4
        pygame.draw.line(screen, white, (self.field_width // 2, 0), (self.field_width // 2, self.field_height), line_width)
        pygame.draw.circle(screen, white, (self.field_width // 2, self.field_height // 2), 50, line_width)
        penalty_width, penalty_height = 150, 300
        pygame.draw.rect(screen, white, (0, (self.field_height - penalty_height) // 2, penalty_width, penalty_height), line_width)
        pygame.draw.rect(screen, white, (self.field_width - penalty_width, (self.field_height - penalty_height) // 2, penalty_width, penalty_height), line_width)
        goal_area_width, goal_area_height = 50, 200
        pygame.draw.rect(screen, white, (0, (self.field_height - goal_area_height) // 2, goal_area_width, goal_area_height), line_width)
        pygame.draw.rect(screen, white, (self.field_width - goal_area_width, (self.field_height - goal_area_height) // 2, goal_area_width, goal_area_height), line_width)

    def draw_goals(self, screen):
        pygame.draw.rect(screen, (255, 255, 255), (*self.left_goal, self.goal_width, self.goal_height))
        pygame.draw.rect(screen, (255, 255, 255), (*self.right_goal, self.goal_width, self.goal_height))
        pygame.draw.circle(screen, (255, 0, 0), (self.left_goal[0] + self.goal_width // 2, self.left_goal[1] + self.goal_height // 2), 5)
        pygame.draw.circle(screen, (255, 0, 0), (self.right_goal[0] + self.goal_width // 2, self.right_goal[1] + self.goal_height // 2), 5)

    def draw_players(self, screen):
        for i, pos in enumerate(self.team_a_positions):
            color = (0, 0, 255) if not self.human_mode or i != self.selected_player else (255, 165, 0)
            pygame.draw.circle(screen, color, (int(pos[0]), int(pos[1])), self.player_size)
        for pos in self.team_b_positions:
            pygame.draw.circle(screen, (255, 0, 0), (int(pos[0]), int(pos[1])), self.player_size)

    def draw_ball(self, screen):
        pygame.draw.circle(screen, (255, 255, 255), (int(self.ball_pos[0]), int(self.ball_pos[1])), self.ball_size)

    def draw_hud(self, screen):
        time_left = max(0, self.max_steps - self.current_step)
        score_text = self.font.render(f"Team A: {self.score_a}  Team B: {self.score_b}", True, (255, 255, 255))
        time_text = self.font.render(f"Time: {time_left}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))
        screen.blit(time_text, (self.field_width - 100, 10))

    def render(self):
        if not self.render_enabled:
            return
        self.draw_field(self.screen)
        self.draw_field_lines(self.screen)
        self.draw_players(self.screen)
        self.draw_ball(self.screen)
        self.draw_goals(self.screen)
        self.draw_hud(self.screen)
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.render_enabled:
            pygame.quit()