import pygame
import numpy as np
import random
import logging
import os

class SoccerEnvironment:
    def __init__(self, render=False, num_players=2, human_mode=False):
        # Set SDL_VIDEODRIVER for macOS compatibility (uncomment if needed)
        # os.environ['SDL_VIDEODRIVER'] = 'quartz'  # or 'x11' if using XQuartz
        self.field_width, self.field_height = 800, 600
        self.goal_width, self.goal_height = 50, 200
        self.player_size = 20
        self.ball_size = 10
        self.render_enabled = render
        self.num_players = num_players
        self.human_mode = human_mode
        self.max_steps = 10000
        self.current_step = 0
        self.score_a = 0
        self.score_b = 0
        self.selected_player = 0
        self.human_action = None
        self.ball_possessor = None
        self.GRAVITY = 0.1  # Scaled down for game scale
        self.last_directions = {'a': [None] * num_players, 'b': [None] * num_players}
        self.kick_cooldown = None
        self.kick_cooldown_duration = 5  # Frames to prevent immediate re-interception
        self.kick_cooldown_counter = 0
        self.last_touch = None  # Track which team last touched the ball ('a' or 'b')
        self.just_reset = True  # Flag to prevent gravity at start

        if render:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.field_width, self.field_height))
            pygame.display.set_caption("Soccer Q-Learning")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("arial", 20)
            logging.info("Pygame initialized for rendering")
        else:
            self.screen = None
            self.clock = None
            self.font = None

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
        self.score_a = 0
        self.score_b = 0
        self.selected_player = 0
        self.human_action = None
        self.ball_possessor = None
        self.last_directions = {'a': [None] * self.num_players, 'b': [None] * self.num_players}
        self.kick_cooldown = None
        self.kick_cooldown_counter = 0
        self.last_touch = None
        self.just_reset = True
        return self.get_state()

    def reset_after_goal(self):
        """Reset ball and players to initial positions after a goal, without resetting scores or time."""
        self.team_a_positions = [[self.field_width // 4, self.field_height // (self.num_players + 1) * (i + 1)] for i in range(self.num_players)]
        self.team_b_positions = [[3 * self.field_width // 4, self.field_height // (self.num_players + 1) * (i + 1)] for i in range(self.num_players)]
        self.ball_pos = [self.field_width // 2, self.field_height // 2]
        self.ball_velocity = [0, 0]
        self.ball_possessor = None
        self.kick_cooldown = None
        self.kick_cooldown_counter = 0
        self.last_directions = {'a': [None] * self.num_players, 'b': [None] * self.num_players}
        self.last_touch = None
        self.just_reset = True
        logging.debug("Field reset after goal")

    def reset_after_out_of_bounds(self, possession_team):
        """Reset ball to a position near the boundary and give possession to the specified team."""
        if self.ball_pos[0] <= self.ball_size:
            self.ball_pos[0] = self.ball_size + 10
        elif self.ball_pos[0] >= self.field_width - self.ball_size:
            self.ball_pos[0] = self.field_width - self.ball_size - 10
        if self.ball_pos[1] <= self.ball_size:
            self.ball_pos[1] = self.ball_size + 10
        elif self.ball_pos[1] >= self.field_height - self.ball_size:
            self.ball_pos[1] = self.field_height - self.ball_size - 10

        self.ball_velocity = [0, 0]
        if possession_team == 'a':
            distances = [((pos[0] - self.ball_pos[0])**2 + (pos[1] - self.ball_pos[1])**2)**0.5 for pos in self.team_a_positions]
            closest_idx = np.argmin(distances)
            self.ball_possessor = ('a', closest_idx)
            self.ball_pos = self.team_a_positions[closest_idx].copy()
        else:
            distances = [((pos[0] - self.ball_pos[0])**2 + (pos[1] - self.ball_pos[1])**2)**0.5 for pos in self.team_b_positions]
            closest_idx = np.argmin(distances)
            self.ball_possessor = ('b', closest_idx)
            self.ball_pos = self.team_b_positions[closest_idx].copy()
        self.kick_cooldown = None
        self.kick_cooldown_counter = 0
        self.just_reset = True
        logging.debug(f"Ball reset after out-of-bounds, possession to {possession_team}")

    def step(self, action_a, action_b):
        self.current_step += 1
        team_a_actions = [action_a] * self.num_players
        team_b_actions = [action_b] * self.num_players
        if self.human_mode:
            human_action = self.get_human_action()
            if human_action is not None:
                team_a_actions[self.selected_player] = human_action

        # Initialize rewards
        reward_a = 0
        reward_b = 0

        speed = 5
        kick_speed = 15
        kick_offset = self.player_size + self.ball_size + 2
        direction_vectors = {
            0: (0, -speed),  # up
            1: (0, speed),   # down
            2: (-speed, 0),  # left
            3: (speed, 0)    # right
        }
        kick_offsets = {
            0: (0, -kick_offset),  # up
            1: (0, kick_offset),   # down
            2: (-kick_offset, 0),  # left
            3: (kick_offset, 0)    # right
        }

        # Update kick cooldown
        if self.kick_cooldown is not None:
            self.kick_cooldown_counter -= 1
            if self.kick_cooldown_counter <= 0:
                self.kick_cooldown = None
                self.kick_cooldown_counter = 0

        # Move Team A players
        for i, action in enumerate(team_a_actions):
            pos = self.team_a_positions[i]
            if action in [0, 1, 2, 3]:
                pos[0] = max(self.player_size, min(self.field_width - self.player_size, pos[0] + direction_vectors[action][0]))
                pos[1] = max(self.player_size, min(self.field_height - self.player_size, pos[1] + direction_vectors[action][1]))
                self.last_directions['a'][i] = action
                player_rect = pygame.Rect(pos[0] - self.player_size, pos[1] - self.player_size, self.player_size * 2, self.player_size * 2)
                ball_rect = pygame.Rect(self.ball_pos[0] - self.ball_size, self.ball_pos[1] - self.ball_size, self.ball_size * 2, self.ball_size * 2)
                if player_rect.colliderect(ball_rect) and self.ball_possessor is None and self.kick_cooldown != ('a', i):
                    self.ball_possessor = ('a', i)
                    self.last_touch = 'a'
                    self.just_reset = False
            elif action == 4:
                if self.ball_possessor == ('a', i):
                    last_action = self.last_directions['a'][i]
                    if last_action is not None:
                        dx, dy = direction_vectors[last_action]
                        self.ball_velocity = [kick_speed * dx / speed, kick_speed * dy / speed]
                        offset_x, offset_y = kick_offsets[last_action]
                        self.ball_pos[0] += offset_x
                        self.ball_pos[1] += offset_y
                    else:
                        dx = self.right_goal[0] + self.goal_width // 2 - self.ball_pos[0]
                        dy = self.right_goal[1] + self.goal_height // 2 - self.ball_pos[1]
                        dist = (dx**2 + dy**2)**0.5
                        if dist > 0:
                            self.ball_velocity = [kick_speed * dx / dist, kick_speed * dy / dist]
                            self.ball_pos[0] += (kick_offset * dx / dist)
                            self.ball_pos[1] += (kick_offset * dy / dist)
                    self.ball_possessor = None
                    self.kick_cooldown = ('a', i)
                    self.kick_cooldown_counter = self.kick_cooldown_duration
                    self.last_touch = 'a'
                    self.just_reset = False
                    logging.debug(f"Team A player {i} kicked ball, velocity: {self.ball_velocity}, pos: {self.ball_pos}")

        # Move Team B players
        for i, action in enumerate(team_b_actions):
            pos = self.team_b_positions[i]
            if action in [0, 1, 2, 3]:
                pos[0] = max(self.player_size, min(self.field_width - self.player_size, pos[0] + direction_vectors[action][0]))
                pos[1] = max(self.player_size, min(self.field_height - self.player_size, pos[1] + direction_vectors[action][1]))
                self.last_directions['b'][i] = action
                player_rect = pygame.Rect(pos[0] - self.player_size, pos[1] - self.player_size, self.player_size * 2, self.player_size * 2)
                ball_rect = pygame.Rect(self.ball_pos[0] - self.ball_size, self.ball_pos[1] - self.ball_size, self.ball_size * 2, self.ball_size * 2)
                if player_rect.colliderect(ball_rect) and self.ball_possessor is None and self.kick_cooldown != ('b', i):
                    self.ball_possessor = ('b', i)
                    self.last_touch = 'b'
                    self.just_reset = False
            elif action == 4:
                if self.ball_possessor == ('b', i):
                    last_action = self.last_directions['b'][i]
                    if last_action is not None:
                        dx, dy = direction_vectors[last_action]
                        self.ball_velocity = [kick_speed * dx / speed, kick_speed * dy / speed]
                        offset_x, offset_y = kick_offsets[last_action]
                        self.ball_pos[0] += offset_x
                        self.ball_pos[1] += offset_y
                    else:
                        dx = self.left_goal[0] + self.goal_width // 2 - self.ball_pos[0]
                        dy = self.left_goal[1] + self.goal_height // 2 - self.ball_pos[1]
                        dist = (dx**2 + dy**2)**0.5
                        if dist > 0:
                            self.ball_velocity = [kick_speed * dx / dist, kick_speed * dy / dist]
                            self.ball_pos[0] += (kick_offset * dx / dist)
                            self.ball_pos[1] += (kick_offset * dy / dist)
                    self.ball_possessor = None
                    self.kick_cooldown = ('b', i)
                    self.kick_cooldown_counter = self.kick_cooldown_duration
                    self.last_touch = 'b'
                    self.just_reset = False
                    logging.debug(f"Team B player {i} kicked ball, velocity: {self.ball_velocity}, pos: {self.ball_pos}")

        # Update ball position if possessed
        if self.ball_possessor is not None:
            team, idx = self.ball_possessor
            pos = self.team_a_positions[idx] if team == 'a' else self.team_b_positions[idx]
            self.ball_pos = [pos[0], pos[1]]
            self.ball_velocity = [0, 0]

        # Update ball physics if not possessed
        else:
            self.ball_pos[0] += self.ball_velocity[0]
            self.ball_pos[1] += self.ball_velocity[1]
            friction = 0.98
            self.ball_velocity[0] *= friction
            self.ball_velocity[1] *= friction
            # Apply gravity only if ball is moving and not just reset
            ball_speed = (self.ball_velocity[0]**2 + self.ball_velocity[1]**2)**0.5
            if ball_speed > 0.1 and not self.just_reset:
                self.ball_velocity[1] += self.GRAVITY
            self.ball_pos[0] = max(self.ball_size, min(self.field_width - self.ball_size, self.ball_pos[0]))
            self.ball_pos[1] = max(self.ball_size, min(self.field_height - self.ball_size, self.ball_pos[1]))

            # Check for ball collision with players (to catch/intercept)
            ball_rect = pygame.Rect(self.ball_pos[0] - self.ball_size * 1.1, self.ball_pos[1] - self.ball_size * 1.1, 
                                self.ball_size * 2.2, self.ball_size * 2.2)
            if ball_speed > 0.5 and self.kick_cooldown_counter == 0:
                collisions = []
                for i, pos in enumerate(self.team_a_positions):
                    if self.kick_cooldown == ('a', i):
                        continue
                    player_rect = pygame.Rect(pos[0] - self.player_size, pos[1] - self.player_size, self.player_size * 2, self.player_size * 2)
                    if ball_rect.colliderect(player_rect):
                        dist = ((pos[0] - self.ball_pos[0])**2 + (pos[1] - self.ball_pos[1])**2)**0.5
                        collisions.append(('a', i, dist))
                for i, pos in enumerate(self.team_b_positions):
                    if self.kick_cooldown == ('b', i):
                        continue
                    player_rect = pygame.Rect(pos[0] - self.player_size, pos[1] - self.player_size, self.player_size * 2, self.player_size * 2)
                    if ball_rect.colliderect(player_rect):
                        dist = ((pos[0] - self.ball_pos[0])**2 + (pos[1] - self.ball_pos[1])**2)**0.5
                        collisions.append(('b', i, dist))
                logging.debug(f"Ball collisions: {[(team, idx, dist) for team, idx, dist in collisions]}, ball_pos: {self.ball_pos}, velocity: {self.ball_velocity}")
                if collisions:
                    team, idx, dist = min(collisions, key=lambda x: x[2])
                    pos = self.team_a_positions[idx] if team == 'a' else self.team_b_positions[idx]
                    self.ball_possessor = (team, idx)
                    self.ball_pos = [pos[0], pos[1]]
                    self.ball_velocity = [0, 0]
                    self.last_touch = team
                    # Reward: +10 for successful pass (teammate catches ball)
                    # Encourages accurate passing within the team
                    if team == self.last_touch and self.last_touch == 'a':
                        reward_a += 10
                    elif team == self.last_touch and self.last_touch == 'b':
                        reward_b += 10
                    logging.debug(f"Ball intercepted by {team} player {idx}, distance: {dist}")

        # Check collisions and handle interception
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
                    if self.ball_possessor is not None and self.kick_cooldown_counter == 0:
                        possessor_team, possessor_idx = self.ball_possessor
                        if (possessor_team == 'a' and possessor_idx == i) or (possessor_team == 'b' and possessor_idx == j):
                            if random.random() < 0.5:
                                if possessor_team == 'a':
                                    self.ball_possessor = ('b', j)
                                    self.ball_pos = [opp_pos[0], opp_pos[1]]
                                    self.last_touch = 'b'
                                else:
                                    self.ball_possessor = ('a', i)
                                    self.ball_pos = [pos[0], pos[1]]
                                    self.last_touch = 'a'
                            self.ball_velocity = [0, 0]

        # Reward shaping
        # Reward: Score-based reward
        # Encourages winning; gives positive reward proportional to score ratio if leading, negative if trailing
        score_ratio_a = self.score_a / (self.score_a + self.score_b) if self.score_a + self.score_b != 0 else 0
        score_ratio_b = self.score_b / (self.score_a + self.score_b) if self.score_a + self.score_b != 0 else 0
        if self.score_a > self.score_b:
            reward_a += 1.0 * score_ratio_a
            reward_b += -1.0 * score_ratio_b
        elif self.score_b > self.score_a:
            reward_a += -1.0 * score_ratio_a
            reward_b += 1.0 * score_ratio_b
        else:
            # Reward: -0.1 for tied score
            # Penalizes stalemate to encourage action
            reward_a += -0.1
            reward_b += -0.1

        # Reward: Proximity to ball
        # Encourages players to stay close to the ball, scaled by distance (closer = higher reward)
        max_field_dist = (self.field_width**2 + self.field_height**2)**0.5
        min_dist_to_ball_a = min((pos[0] - self.ball_pos[0])**2 + (pos[1] - self.ball_pos[1])**2 for pos in self.team_a_positions)**0.5
        min_dist_to_ball_b = min((pos[0] - self.ball_pos[0])**2 + (pos[1] - self.ball_pos[1])**2 for pos in self.team_b_positions)**0.5
        reward_a += 1.0 * (1 - min_dist_to_ball_a / max_field_dist)
        reward_b += 1.0 * (1 - min_dist_to_ball_b / max_field_dist)

        # Reward: Defensive positioning
        # Encourages players to stay near their own goal when the ball is in their half
        ball_in_a_half = self.ball_pos[0] < self.field_width / 2
        ball_in_b_half = self.ball_pos[0] > self.field_width / 2
        for pos in self.team_a_positions:
            dist_to_own_goal = ((pos[0] - self.left_goal[0])**2 + (pos[1] - self.left_goal[1])**2)**0.5
            if ball_in_a_half and dist_to_own_goal < 150:
                reward_a += 2.0  # Reward: +2 for being near own goal when defending
        for pos in self.team_b_positions:
            dist_to_own_goal = ((pos[0] - self.right_goal[0])**2 + (pos[1] - self.right_goal[1])**2)**0.5
            if ball_in_b_half and dist_to_own_goal < 150:
                reward_b += 2.0  # Reward: +2 for being near own goal when defending

        # Reward: Ball progression
        # Encourages moving the ball toward the opponent's goal
        ball_progress_a = 1 - self.ball_pos[0] / self.field_width
        ball_progress_b = self.ball_pos[0] / self.field_width
        reward_a += 5.0 * ball_progress_a  # Reward: Up to +5 for ball closer to opponent's goal
        reward_b += 5.0 * ball_progress_b  # Reward: Up to +5 for ball closer to opponent's goal

        # Reward: Proximity to opponent's goal
        # Encourages positioning the ball near the opponent's goal
        dist_ball_to_right_goal = ((self.ball_pos[0] - (self.right_goal[0] + self.goal_width // 2))**2 + 
                                (self.ball_pos[1] - (self.right_goal[1] + self.goal_height // 2))**2)**0.5
        dist_ball_to_left_goal = ((self.ball_pos[0] - (self.left_goal[0] + self.goal_width // 2))**2 + 
                                (self.ball_pos[1] - (self.left_goal[1] + self.goal_height // 2))**2)**0.5
        if dist_ball_to_right_goal < 100:
            reward_a += 5.0  # Reward: +3 for ball near opponent's goal (right goal for Team A)
        if dist_ball_to_left_goal < 100:
            reward_b += 5.0  # Reward: +3 for ball near opponent's goal (left goal for Team B)

        # Reward: Strategic positioning with possession
        # Encourages players with the ball to be near the goal or pass to better-positioned teammates
        for i, pos in enumerate(self.team_a_positions):
            player_rect = pygame.Rect(pos[0] - self.player_size, pos[1] - self.player_size, self.player_size * 2, self.player_size * 2)
            if self.ball_possessor == ('a', i):
                dist_to_goal = ((pos[0] - self.right_goal[0])**2 + (pos[1] - self.right_goal[1])**2)**0.5
                if dist_to_goal < 100:
                    reward_a += 5  # Reward: +5 for possessing ball near opponent's goal
                else:
                    for j, teammate_pos in enumerate(self.team_a_positions):
                        if i != j:
                            teammate_dist_to_goal = ((teammate_pos[0] - self.right_goal[0])**2 + (teammate_pos[1] - self.right_goal[1])**2)**0.5
                            if teammate_dist_to_goal < dist_to_goal:
                                reward_a += 2  # Reward: +2 for having a teammate closer to goal (encourages passing)

        # Reward: Team spacing
        # Encourages players to maintain optimal distances from teammates (not too close, not too far)
        for i, pos in enumerate(self.team_a_positions):
            for j, teammate_pos in enumerate(self.team_a_positions):
                if i < j:
                    dist = ((pos[0] - teammate_pos[0])**2 + (pos[1] - teammate_pos[1])**2)**0.5
                    if 50 < dist < 200:
                        reward_a += 0.5  # Reward: +0.5 for ideal teammate spacing
                    elif dist < 50:
                        reward_a -= 0.5  # Reward: -0.5 for teammates too close (overcrowding)

        # Reward: Avoiding opponents
        # Encourages players to maintain distance from opponents
        for pos in self.team_a_positions:
            min_dist_to_opponent = min((pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 for opp_pos in self.team_b_positions)**0.5
            if min_dist_to_opponent > 100:
                reward_a += 1  # Reward: +1 for staying far from opponents

        # Reward: Strategic positioning with possession (Team B)
        # Same as Team A, encourages ball possession near goal or passing to better-positioned teammates
        for i, pos in enumerate(self.team_b_positions):
            if self.ball_possessor == ('b', i):
                dist_to_goal = ((pos[0] - self.left_goal[0])**2 + (pos[1] - self.left_goal[1])**2)**0.5
                if dist_to_goal < 100:
                    reward_b += 5  # Reward: +5 for possessing ball near opponent's goal
                else:
                    for j, teammate_pos in enumerate(self.team_b_positions):
                        if i != j:
                            teammate_dist_to_goal = ((teammate_pos[0] - self.left_goal[0])**2 + (teammate_pos[1] - self.left_goal[1])**2)**0.5
                            if teammate_dist_to_goal < dist_to_goal:
                                reward_b += 2  # Reward: +2 for having a teammate closer to goal (encourages passing)

        # Reward: Team spacing (Team B)
        # Encourages optimal teammate distances
        for i, pos in enumerate(self.team_b_positions):
            for j, teammate_pos in enumerate(self.team_b_positions):
                if i < j:
                    dist = ((pos[0] - teammate_pos[0])**2 + (pos[1] - teammate_pos[1])**2)**0.5
                    if 50 < dist < 200:
                        reward_b += 0.5  # Reward: +0.5 for ideal teammate spacing
                    elif dist < 50:
                        reward_b -= 0.5  # Reward: -0.5 for teammates too close (overcrowding)

        # Reward: Avoiding opponents (Team B)
        # Encourages staying away from opponents
        for pos in self.team_b_positions:
            min_dist_to_opponent = min((pos[0] - opp_pos[0])**2 + (pos[1] - opp_pos[1])**2 for opp_pos in self.team_a_positions)**0.5
            if min_dist_to_opponent > 100:
                reward_b += 1  # Reward: +1 for staying far from opponents

        # Handle out-of-bounds
        out_of_bounds = (self.ball_pos[0] <= self.ball_size or 
                        self.ball_pos[0] >= self.field_width - self.ball_size or
                        self.ball_pos[1] <= self.ball_size or 
                        self.ball_pos[1] >= self.field_height - self.ball_size)
        if out_of_bounds:
            possession_team = 'b' if self.last_touch == 'a' else 'a'
            self.reset_after_out_of_bounds(possession_team)
            # Reward: -50 for ball going out of bounds
            # Penalizes losing control of the ball
            reward_a -= 50
            reward_b -= 50
            logging.debug(f"Ball out of bounds, possession to {possession_team}")

        # Goal and termination
        done = False
        right_goal_rect = pygame.Rect(self.right_goal[0], self.right_goal[1], self.goal_width, self.goal_height)
        left_goal_rect = pygame.Rect(self.left_goal[0], self.left_goal[1], self.goal_width, self.goal_height)

        if ball_rect.colliderect(right_goal_rect):
            self.score_a += 1
            # Reward: +50000 for scoring a goal, -50000 for opponent conceding
            # Strongly incentivizes scoring
            reward_a += 50000
            reward_b += -50000
            self.reset_after_goal()
            logging.debug(f"Goal scored by Team A, score: {self.score_a}-{self.score_b}")
        elif ball_rect.colliderect(left_goal_rect):
            self.score_b += 1
            # Reward: +50000 for scoring a goal, -50000 for opponent conceding
            # Strongly incentivizes scoring
            reward_a += -50000
            reward_b += 50000
            self.reset_after_goal()
            logging.debug(f"Goal scored by Team B, score: {self.score_a}-{self.score_b}")
        elif self.current_step >= self.max_steps:
            done = True

        next_state = self.get_state()
        return next_state, reward_a, reward_b, done

    def get_state(self):
        state = []
        for pos in self.team_a_positions:
            state.extend([pos[0], pos[1]])
        for pos in self.team_b_positions:
            state.extend([pos[0], pos[1]])
        state.extend([self.ball_pos[0], self.ball_pos[1], self.ball_velocity[0], self.ball_velocity[1]])
        state.extend([self.right_goal[0] + self.goal_width // 2, self.right_goal[1] + self.goal_height // 2])
        #print(f"State: {state}")
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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise SystemExit
        self.draw_field(self.screen)
        self.draw_field_lines(self.screen)
        self.draw_players(self.screen)
        self.draw_ball(self.screen)
        self.draw_goals(self.screen)
        self.draw_hud(self.screen)
        pygame.display.flip()
        pygame.display.update()
        self.clock.tick(60)
        logging.debug("Rendered frame")

    def close(self):
        if self.render_enabled:
            pygame.quit()
            logging.info("Pygame quit")