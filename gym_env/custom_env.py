import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import matplotlib.backends.backend_agg as agg

class CustomEnv(gym.Env):
    metadata = {'render_modes': ['human','rgb_array']}

    def __init__(self,render_mode='human', num_objects=2):
        super(CustomEnv, self).__init__()
        self.grid_size = (10, 10)
        self.action_space = spaces.Discrete(5)  # 0: Up, 1: Down, 2: Left, 3: Right, 4: Interact
        self.observation_space = spaces.Box(low=0, high=5, shape=(10, 10), dtype=np.int32)

        self.robot_code = 1
        self.wall_code = 2
        self.movable_code = 3
        self.non_movable_code = 4
        self.goal_code = 5
        self.colors = {0: 'white', 1: 'orange', 2: 'grey', 3: 'blue', 4: 'red', 5: 'green'}

        self.num_objects = num_objects
        self.environment_setup = {'state': None, 'robot_pos': None, 'room_top_left': None, 'orientation': 'up', 'digital_mind': {}}
        self.orientation = 'up'
        self.previous_action = None
        self.action_observation_log = []
        self.render_mode = render_mode
        self.setup_environment()

    def setup_environment(self):
        self.state = np.zeros(self.grid_size, dtype=np.int32)
        self._place_fixed_walls_and_room()
        self._place_objects()
        self._place_goal()
        self.robot_movement_state = np.copy(self.state)
        self.environment_setup['state'] = np.copy(self.state)
        self.environment_setup['robot_movement_state'] =self.robot_movement_state  # Store initial robot movement state
        self.environment_setup['robot_pos'] = [2, 1]
        # self.environment_setup['room_top_left'] = self.room_top_left
        self.environment_setup['digital_mind'] = np.full(self.grid_size, '', dtype=object)  # Initialize digital mind

    def reset(self):
        self.setup_environment()
        return self.state

    def _place_fixed_walls_and_room(self):
        self.state[0, :] = self.wall_code
        self.state[-1, :] = self.wall_code
        self.state[:, 0] = self.wall_code
        self.state[:, -1] = self.wall_code

        room_placed = True # Removing room for now
        while not room_placed:
            self.room_top_left = (np.random.randint(1, 20), np.random.randint(1, 20))
            # Check if the proposed opening is adjacent to at least one free space
            opening_adjacent_cells = [(self.room_top_left[0] + 3, self.room_top_left[1] - 1),
                                    (self.room_top_left[0] + 3, self.room_top_left[1] + 1)]
            if any(self.state[cell] == 0 for cell in opening_adjacent_cells):
                room_placed = True
                for i in range(self.room_top_left[0], self.room_top_left[0] + 5):
                    for j in range(self.room_top_left[1], self.room_top_left[1] + 5):
                        if i == self.room_top_left[0] + 3 and j == self.room_top_left[1]:
                            continue  # Leave a space for the door
                        if i in {self.room_top_left[0], self.room_top_left[0] + 4} or j in {self.room_top_left[1], self.room_top_left[1] + 4}:
                            self.state[i, j] = self.wall_code

    def _place_objects(self):
        free_cells = [(r, c) for r in range(1, 9) for c in range(1, 9) if self.state[r, c] == 0]
        np.random.shuffle(free_cells)
        for i in range(self.num_objects):
            obj_type = self.movable_code if i % 2 == 0 else self.non_movable_code
            cell = free_cells.pop()
            self.state[cell] = obj_type

    def _place_goal(self):
        free_cells = [(r, c) for r in range(1, 9) for c in range(1, 9) if self.state[r, c] == 0]
        np.random.shuffle(free_cells)
        for cell in free_cells:
            r, c = cell
            adjacent_free_cells = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
            adjacent_free_cells = [cell for cell in adjacent_free_cells if self.state[cell] == 0]
            if len(adjacent_free_cells) >= 4:
                self.state[r, c] = self.goal_code
                break
        
    def update_orientation(self, action):
        # Map actions to orientations
        orientations = ['up', 'down', 'left', 'right']
        if action in [0, 1, 2, 3]:  # Check if action corresponds to a movement
            self.orientation = orientations[action]

    def step(self, action):
        done = False
        reward = 0
        self.state = np.copy(self.environment_setup['state'])
        self.robot_pos = self.environment_setup['robot_pos']
        next_pos = self.get_fov_cell(self.robot_pos, action)
        cell_code = self.get_fov_info(next_pos)

        # Movement actions
        if action < 4:
            if cell_code in [self.wall_code, self.non_movable_code]:
                reward -= 2  # Penalize collision
                next_pos, next_action = self.find_alternative_action_based_on_movement_state(self.robot_pos, action, threshold=2)
                cell_code = self.get_fov_info(next_pos)
                self.update_orientation(next_action)
                self.update_robot_movement_state(self.robot_pos, next_pos, cell_code)
                self.robot_pos = next_pos
                self.previous_action = next_action
            elif cell_code == self.movable_code:
                reward -= 1
                self.previous_action = 4
            else:
                # Reward for moving to a new cell, promoting local exploration
                if self.robot_movement_state[tuple(next_pos)] == 0:
                    reward += 1  # Reward for visiting a new cell for the first time
                self.update_robot_movement_state(self.robot_pos, next_pos, cell_code)
                self.robot_pos = next_pos
                self.previous_action = action
                self.update_orientation(action)

            if self.state[tuple(self.robot_pos)] == self.goal_code:
                done = True
                reward += 10

        elif action == 4:
            self.robot_movement_state[tuple(self.robot_pos)] += 1
            moved_object, interaction_reward = self.move_object_if_possible()
            reward += interaction_reward
            if not moved_object:
                reward = -1

        self.update_state_and_mind()
        return self.state, reward, done, {}

    def update_robot_movement_state(self, old_pos, new_pos, cell_code):
        # If moving to a new cell, increment its value to indicate it's been visited
        print(f'Updating Robot Movement State : {old_pos} -> {new_pos}, Code : {cell_code}')
        if old_pos != new_pos:
            self.robot_movement_state[tuple(new_pos)] += 1
        
        # If the new position is a wall or immovable object, discourage revisiting this path
        # by artificially increasing the visit count. This logic might need adjustments.
        if cell_code in [self.wall_code, self.non_movable_code]:
            self.robot_movement_state[tuple(new_pos)] *= 2  # Example adjustment

    def find_alternative_action_based_on_movement_state(self, robot_pos, action, threshold=2):
        print(f"Initial robot position: {robot_pos}, action: {action}, orientation: {self.orientation}")
        
        # Mapping from orientations to potential actions to take next
        orientation_to_direction = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        current_orientation_index = orientation_to_direction[self.orientation]
        print(f"Current orientation index: {current_orientation_index}")

        directions = [(-1, 0),(1, 0), (0, -1), (0, 1)]  # Mappings for Up, Down, Left, Right
        alternatives = []

        for idx, (dx, dy) in enumerate(directions):
            next_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
            if self.is_out_of_bounds(next_pos) or self.state[next_pos] in [self.wall_code, self.non_movable_code]:
                print(f"Skipping direction {idx} due to out-of-bounds or blocked path at position {next_pos} from {robot_pos}")
                continue
            visits = self.robot_movement_state[next_pos]
            alternatives.append((idx, visits))

        print(f"Alternatives before sorting: {alternatives}")
        alternatives.sort(key=lambda x: x[1])
        print(f"Alternatives after sorting: {alternatives}")

        if alternatives:
            for alt in alternatives:
                if alt[1] < threshold:
                    best_action = alt[0]
                    dx, dy = directions[best_action]
                    next_action_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
                    print(f"Selected alternative action: {best_action}, resulting position: {next_action_pos}")
                    return next_action_pos, best_action

        # Use orientation to decide on a new action intelligently
        new_orientations = [(current_orientation_index + i) % 4 for i in range(1, 5)]
        for new_orientation_index in new_orientations:
            dx, dy = directions[new_orientation_index]
            next_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
            if not self.is_out_of_bounds(next_pos) and self.state[next_pos] not in [self.wall_code, self.non_movable_code]:
                new_action = new_orientation_index
                self.orientation = ['up', 'down', 'left', 'right'][new_action]  # Update orientation
                print(f"New action based on orientation: {new_action}, new position: {next_pos}")
                return next_pos, new_action

        print(f"No viable alternative found, staying in place. Position: {robot_pos}, action: {action}")
        return robot_pos, action  # Stay in place if no viable option is found


    def is_out_of_bounds(self, pos):
        r, c = pos
        return not (0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1])
    
    def change_action_if_collision_is_nearby(self, robot_pos, action):
        # Directions: 0: Up, 1: Down, 2: Left, 3: Right
        alternative_actions = [2, 3] if action in [0, 1] else [0, 1]  # Choose sideways actions based on current action
        for alt_action in alternative_actions:
            next_pos = self.get_fov_cell(robot_pos, alt_action)
            if not self.is_out_of_bounds(next_pos) and self.state[tuple(next_pos)] in [0, self.goal_code]:
                return next_pos,alt_action  # Return the alternative action if next position is free or goal
        return robot_pos,action  # Return original action if no alternatives are found
    
    def move_object_if_possible(self):
        direction = {'up': 0, 'down': 1, 'left': 2, 'right': 3}[self.orientation]
        fov_cell = self.get_fov_cell(self.robot_pos, direction)
        next_cell = self.get_fov_cell(fov_cell, direction)

        if self.environment_setup['digital_mind'][tuple(fov_cell)] == 'moved':
            return False, -1  # Object already moved

        if self.state[tuple(fov_cell)] == self.movable_code and self.state[tuple(next_cell)] in [0, self.goal_code]:
            # Clear the current cell and move the object to the next cell
            self.state[tuple(fov_cell)] = 0
            self.state[tuple(next_cell)] = self.movable_code
            self.robot_pos = fov_cell  # Move the robot to the position previously occupied by the object
            self.environment_setup['digital_mind'][tuple(fov_cell)] = 'moved'
            self.previous_action = direction
            self.update_orientation(self.orientation)
            
            # Update robot movement state to reflect successful interaction
            self.robot_movement_state[tuple(next_cell)] += 1
            
            return True, 1
        return False, 0

    def update_state_and_mind(self):
        self.state = np.copy(self.environment_setup['state'])
        self.state[tuple(self.robot_pos)] = self.robot_code  # Update robot's position in the state
        self.environment_setup['robot_pos'] = self.robot_pos  # Update robot's position in the environment setup
        self.environment_setup['orientation'] = self.orientation  # Update orientation in the environment setup

    def get_fov_info(self, pos):
        # Return the cell code for the FOV cell or 'OB' if out of bounds
        if 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]:
            return self.state[tuple(pos)]
        else:
            return 'OB'
    
    def get_fov_cell(self, robot_pos, action):
        # Calculate the FOV based on the current position and direction
        if action == 0:  # Up
            return [max(robot_pos[0] - 1, 0), robot_pos[1]]
        elif action == 1:  # Down
            return [min(robot_pos[0] + 1, self.grid_size[0] - 1), robot_pos[1]]
        elif action == 2:  # Left
            return [robot_pos[0], max(robot_pos[1] - 1, 0)]
        elif action == 3:  # Right
            return [robot_pos[0], min(robot_pos[1] + 1, self.grid_size[1] - 1)]
        return robot_pos

    def log_action_observation(self, action, observation, robot_pos):
        self.action_observation_log.append({
            'action': action,
            'observation': observation,
            'robot_pos': robot_pos
        })

    def render(self, mode=None):
        if mode != None:
            if mode == 'human':
                fig, ax = plt.subplots()
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)
                plt.grid()
                for r in range(10):
                    for c in range(10):
                        color = self.colors[self.state[r, c]]
                        rect = Rectangle((c, 9-r), 1, 1, color=color)
                        ax.add_patch(rect)
                plt.show()
            elif mode == 'rgb_array':
                fig, ax = plt.subplots()
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)
                plt.grid()
                for r in range(10):
                    for c in range(10):
                        color = self.colors[self.state[r, c]]
                        rect = Rectangle((c, 9-r), 1, 1, color=color)
                        ax.add_patch(rect)
                
                # Convert the Matplotlib figure to an RGB array
                canvas = agg.FigureCanvasAgg(fig)
                canvas.draw()
                width, height = fig.get_size_inches() * fig.get_dpi()
                image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
                
                plt.close(fig)  # Close the figure to prevent it from being displayed
                return image
        else:
            if self.render_mode == 'human':
                fig, ax = plt.subplots()
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)
                plt.grid()
                for r in range(10):
                    for c in range(10):
                        color = self.colors[self.state[r, c]]
                        rect = Rectangle((c, 9-r), 1, 1, color=color)
                        ax.add_patch(rect)
                plt.show()
            elif self.render_mode == 'rgb_array':
                fig, ax = plt.subplots()
                ax.set_xlim(0, 10)
                ax.set_ylim(0, 10)
                plt.grid()
                for r in range(10):
                    for c in range(10):
                        color = self.colors[self.state[r, c]]
                        rect = Rectangle((c, 9-r), 1, 1, color=color)
                        ax.add_patch(rect)
                
                # Convert the Matplotlib figure to an RGB array
                canvas = agg.FigureCanvasAgg(fig)
                canvas.draw()
                width, height = fig.get_size_inches() * fig.get_dpi()
                image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
                
                plt.close(fig)  # Close the figure to prevent it from being displayed
                return image
