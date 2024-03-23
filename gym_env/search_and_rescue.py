import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import matplotlib.backends.backend_agg as agg
import copy
import os 
import pickle
from utils.helper import generate_maze_with_objects,visualisemaze,check_and_create_directory
import wandb
import logging
import math
import pdb
import time 

class SearchAndRescueEnv(gym.Env):
    metadata = {'render_modes': ['human','rgb_array']}

    def __init__(
            self,
            render_mode:str='human',
            num_movable_objects:int=0,
            num_immovable_objects:int=0,
            num_start_pos:int=1,
            grid_size:int=20, # Will be a Square
            max_step:int=400,
            randomness:bool=False,
            save_map:bool=True,
            log_dir:str='logs'
        ):
        super(SearchAndRescueEnv, self).__init__()
        self.use_random = randomness
        self.max_steps = max_step
        self.log_dir = check_and_create_directory(log_dir)
        self.grid_size = (grid_size,grid_size)
        self.num_m = num_movable_objects
        self.num_i = num_immovable_objects
        self.num_s = num_start_pos
        self.MAP_PLAN,self.M_KB,self.U_KB = generate_maze_with_objects(grid_size,grid_size,num_movable_objects,num_immovable_objects,num_start_pos)
        if save_map:
            with open(os.path.join(log_dir,'maze_plan.pkl'),'wb') as file:
                pickle.dump(self.MAP_PLAN,file)
            visualisemaze(self.MAP_PLAN,log_dir)
        self.episode_count = 0
        self.num_objects = num_movable_objects + num_immovable_objects
        # Code assignments
        self.robot_code = 5
        self.wall_code = 0
        self.movable_code = 2
        self.non_movable_code = 3
        self.goal_code = 6
        self.room_code = 4
        self.free_space = 1
        self.colors = {1: 'white', 5: 'orange', 0: 'grey', 2: 'blue', 3: 'red', 6: 'green',4:'grey'}
        
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right 
        self.observation_space = spaces.Box(low=0, high=6, shape=self.grid_size, dtype=np.int32)

        self.environment_setup = {'state': None, 'robot_pos': None, 'robot_movement_state': None}
        self.render_mode = render_mode
        self.state = None
        self.reset()
        self.setup_environment()
    
    def get_robot_position(self)->np.ndarray:
        return np.argwhere(self.state == self.robot_code)[0]
    
    def get_goal_position(self)->np.ndarray:
        return np.argwhere(self.state==self.goal_code)[0]

    def setup_environment(self):
        self.robot_movement_state = np.ones(self.grid_size,dtype=np.int32)
        self.environment_setup['state'] = np.copy(self.state)
        self.environment_setup['robot_movement_state'] =self.robot_movement_state  # Store initial robot movement state
        self.environment_setup['robot_pos'] = self.get_robot_position()

    def reset(self):
        if self.use_random:
            self.MAP_PLAN,self.M_KB,self.U_KB = generate_maze_with_objects(self.grid_size[0],self.grid_size[1],self.num_m,self.num_i,self.num_s)
            map_log_dir = check_and_create_directory(os.path.join(self.log_dir,"maps",str(self.episode_count)))
            with open(os.path.join(map_log_dir,'maze_plan.pkl'),'wb') as file:
                pickle.dump(self.MAP_PLAN,file)
            visualisemaze(self.MAP_PLAN,map_log_dir)
        else:
            self.state = np.copy(self.M_KB)
        self.cumulative_reward = 0
        self.current_step = 0
        self.cumulative_immovable_interactions = 0
        self.cumulative_movable_interactions = 0
        
        # First Experiment 
        """
        Return the MAP_STATE as the Observation space - this means we are saying the causal movability is determined wth texture and the robot knows about it
        """
        # self.state = np.copy(self.M_KB)
        self.state = np.ones(self.grid_size,dtype=np.int32)
        robot_pos = np.argwhere(self.M_KB == self.robot_code)[0]
        goal_pos = np.argwhere(self.M_KB == self.goal_code)[0]
        self.state[tuple(robot_pos)] = self.robot_code
        self.state[tuple(goal_pos)] = self.goal_code
        # Second Experiment 
        """
        Returns the State as the Unknwon KB array - this means the agent without knowledge about movability But self.M_KB will be used to govern the translation of actions
        """
        # self.state = self.U_KB
        # Third Experiment
        """
        Returns the State as the Unknown KB Array - this means the causal agent will be updating movability each iteration forming a relation and then assigning the state . Then Self.M_KB becomes our evaluation 
        """
        # self.start  = self.U_KB
        self.setup_environment()
        self.robot_pos = self.get_robot_position()
        
        return self.state
            
    def get_fov_info(self, pos):
        if 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]:
            return self.M_KB[tuple(pos)] 
        else:
            return 'OB'
    
    def get_fov_cell(self, robot_pos, action):
        if action == 0:  # Up
            return [max(robot_pos[0] - 1, 0), robot_pos[1]]
        elif action == 1:  # Down
            return [min(robot_pos[0] + 1, self.grid_size[0] - 1), robot_pos[1]]
        elif action == 2:  # Left
            return [robot_pos[0], max(robot_pos[1] - 1, 0)]
        elif action == 3:  # Right
            return [robot_pos[0], min(robot_pos[1] + 1, self.grid_size[1] - 1)]
        return robot_pos

    def update_robot_position_in_state(self,old_pos,next_pos):
        self.state[tuple(old_pos)] = self.free_space
        self.state[tuple(next_pos)] = self.robot_code
    
    def update_robot_movement_state(self, old_pos, new_pos, cell_code):
        logging.info(f'Updating Robot Movement State : {old_pos} -> {new_pos}, Code : {cell_code}')
        
        if tuple(old_pos) != tuple(new_pos):
            self.robot_movement_state[tuple(new_pos)] += 1
        
        if cell_code in [self.wall_code,self.room_code, self.non_movable_code]:
            self.robot_movement_state[tuple(new_pos)] *= 2  # Scale higher visit for objects
    
    def translate_action(self,action,current_pos,next_pos,next_cell_code):
        if next_cell_code not in [self.wall_code, self.non_movable_code, self.room_code,self.movable_code,self.goal_code]: # If Free Space
            print('In free space translate')
            self.update_robot_position_in_state(current_pos,next_pos)
            self.update_robot_movement_state(current_pos,next_pos,next_cell_code)
            action_info = {
                "object_type":1,
                "movablity":True,
                "has_reached_goal":False,
                "is_freespace":True
            }
        elif next_cell_code in [self.wall_code,self.room_code, self.non_movable_code]:
            self.update_robot_position_in_state(current_pos,current_pos)
            self.update_robot_movement_state(current_pos,current_pos,next_cell_code)
            self.state[tuple(next_pos)] = next_cell_code # updates observation
            if next_cell_code == self.non_movable_code:
                self.cumulative_immovable_interactions += 1
            action_info = {
                "object_type":next_cell_code,
                "movablity":False,
                "has_reached_goal":False,
                "is_freespace":False
            }
        elif next_cell_code == self.movable_code: # movable object
            self.cumulative_movable_interactions += 1
            new_obj_pos = self.get_fov_cell(next_pos, action)
            new_obj_pos_fov_cell_code = self.get_fov_info(new_obj_pos)
            if new_obj_pos_fov_cell_code not in [self.non_movable_code,self.wall_code,self.room_code,self.movable_code]:
                self.update_robot_position_in_state(current_pos,next_pos)
                self.update_robot_movement_state(current_pos,next_pos,next_cell_code)
                self.state[tuple(new_obj_pos)] = 2
                action_info = {
                    "object_type":next_cell_code,
                    "movablity":True,
                    "has_reached_goal":False,
                    "is_freespace":False
                }
            else:
                self.update_robot_position_in_state(current_pos,current_pos)
                self.update_robot_movement_state(current_pos,current_pos,self.non_movable_code) # manually updating visit for this state to be higher so it doesnt visit it
                self.state[tuple(next_pos)] = next_cell_code
                self.state[tuple(new_obj_pos)] = new_obj_pos_fov_cell_code
                action_info = {
                    "object_type":next_cell_code,
                    "movablity":False,
                    "has_reached_goal":False,
                    "is_freespace":False
                }
        else: # next cell code is goal
            self.update_robot_position_in_state(current_pos,next_pos)
            self.state[tuple(next_pos)] = next_cell_code
            action_info = {
                "object_type":next_cell_code,
                "movablity":True,
                "has_reached_goal":True,
                "is_freespace":True
            }
            
        return action_info

    def get_maximum_distance(self):
        return math.sqrt(self.grid_size[0]**2 + self.grid_size[1]**2)
    
    def calculate_reward(self,next_pos,next_cell_code) :
        reward = 0
        done = False
        
        if next_cell_code in [self.wall_code,self.room_code]: # penalize for collision with wall and room
            reward -= 1
            # done = True
        
        if self.robot_movement_state[tuple(next_pos)] == 0: # reward for new visits - promotes exploration
            reward += 2
        
        if next_cell_code == self.goal_code: # reward for reaching goal higher
            reward += 10
            done = True

        # goal_pos = self.get_goal_position()
        # max_dist = self.get_maximum_distance()
        # distance = math.sqrt((next_pos[0] - goal_pos[0])**2 + (next_pos[1] - goal_pos[1])**2)
        # distance_reward = 5 * (1 - (distance / max_dist))
        # reward += max(distance_reward, 0)
        
        if self.current_step >= self.max_steps:
            done = True
        
        return reward,done
    
    def log_interactions(self,next_cell_code):
        if next_cell_code == self.non_movable_code:
            wandb.log({"immovable_interactions":self.cumulative_immovable_interactions})
        elif next_cell_code == self.movable_code:
            wandb.log({"movable_interactions":self.cumulative_movable_interactions})
            
    def step(self, action):
        done = False
        self.current_step += 1
        self.robot_pos = self.get_robot_position()
        next_pos = self.get_fov_cell(self.robot_pos, action)
        cell_code = self.get_fov_info(next_pos)
        reward,done = self.calculate_reward(next_pos,cell_code)
        self.translate_action(action,self.robot_pos,next_pos,cell_code)
        self.cumulative_reward += reward        
        if done:
            self.episode_count += 1
        
        wandb.log({"episode":self.episode_count,"cummulative_reward":self.cumulative_reward})
        wandb.log({f"episode-{self.episode_count}":self.current_step,"reward":reward})
        self.log_interactions(cell_code)
        
        return self.state, reward, done, {}

    def render(self, mode=None):
        if mode != None:
            if mode == 'human':
                fig, ax = plt.subplots()
                ax.set_xlim(0, self.grid_size[0])
                ax.set_ylim(0, self.grid_size[1])
                plt.grid()
                for r in range(self.grid_size[0]):
                    for c in range(self.grid_size[1]):
                        color = self.colors[self.state[r, c]]
                        rect = Rectangle((c, (self.grid_size[0]-1)-r), 1, 1, color=color)
                        ax.add_patch(rect)
                plt.show()
            elif mode == 'rgb_array':
                fig, ax = plt.subplots()
                ax.set_xlim(0, self.grid_size[0])
                ax.set_ylim(0, self.grid_size[1])
                plt.grid()
                for r in range(self.grid_size[0]):
                    for c in range(self.grid_size[1]):
                        color = self.colors[self.state[r, c]]
                        rect = Rectangle((c, (self.grid_size[0]-1)-r), 1, 1, color=color)
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
                ax.set_xlim(0, self.grid_size[0])
                ax.set_ylim(0, self.grid_size[1])
                plt.grid()
                for r in range(self.grid_size[0]):
                    for c in range(self.grid_size[1]):
                        color = self.colors[self.state[r, c]]
                        rect = Rectangle((c, (self.grid_size[0]-1)-r), 1, 1, color=color)
                        ax.add_patch(rect)
                plt.show()
            elif self.render_mode == 'rgb_array':
                fig, ax = plt.subplots()
                ax.set_xlim(0, self.grid_size[0])
                ax.set_ylim(0, self.grid_size[1])
                plt.grid()
                for r in range(self.grid_size[0]):
                    for c in range(self.grid_size[1]):
                        color = self.colors[self.state[r, c]]
                        rect = Rectangle((c, (self.grid_size[0]-1)-r), 1, 1, color=color)
                        ax.add_patch(rect)
                
                # Convert the Matplotlib figure to an RGB array
                canvas = agg.FigureCanvasAgg(fig)
                canvas.draw()
                width, height = fig.get_size_inches() * fig.get_dpi()
                image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
                
                plt.close(fig)  # Close the figure to prevent it from being displayed
                return image
