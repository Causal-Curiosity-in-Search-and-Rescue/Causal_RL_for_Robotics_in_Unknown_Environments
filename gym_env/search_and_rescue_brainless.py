import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import random
import matplotlib.backends.backend_agg as agg
import copy
import os 
import pickle
from utils.helper import generate_maze_with_objects,visualisemaze,check_and_create_directory,read_config,load_saved_map
import wandb
import logging
import math
import pdb
import time 

"""
Using RL Observation Space - Where Object Movability is not shown in Observation Space
"""
class SearchAndRescueNoCausalEnv(gym.Env):
    metadata = {'render_modes': ['human','rgb_array']}

    def __init__(
            self,
            render_mode:str='human'
        ):
        super(SearchAndRescueNoCausalEnv, self).__init__()
        CONFIG = read_config()
        ENV_CONFIG = CONFIG['environment']
        self.use_random = ENV_CONFIG['randomness']
        self.max_steps = ENV_CONFIG['max_steps']
        self.log_dir = check_and_create_directory( CONFIG['log_dir'])
        self.grid_size = ( ENV_CONFIG['grid_size'], ENV_CONFIG['grid_size'])
        self.num_m =  ENV_CONFIG['num_movable_objects']
        self.num_i =  ENV_CONFIG['num_immovable_objects']
        self.num_s =  ENV_CONFIG['num_start_pos']
        self.env_config = ENV_CONFIG
        if CONFIG['mode'] == 'train':
            self.MAP_PLAN,self.M_KB,self.U_KB = generate_maze_with_objects(self.grid_size[0],self.grid_size[1],self.num_m,self.num_i,self.num_s)
            if  ENV_CONFIG['save_map']:
                with open(os.path.join(CONFIG['log_dir'],'maze_plan.pkl'),'wb') as file:
                    pickle.dump(self.MAP_PLAN,file)
                visualisemaze(self.MAP_PLAN,self.log_dir)
        elif CONFIG['mode'] == 'test':
            self.MAP_PLAN,self.M_KB,self.U_KB = load_saved_map(CONFIG["inference"]["map_path"])
        self.episode_count = 0
        self.num_objects = self.num_m + self.num_i
        # Code assignments
        self.robot_code = 5
        self.wall_code = 0
        self.movable_code = 2
        self.non_movable_code = 3
        self.object_code = 2 # Will be used for Non Causal Observation Space so the Agent will not distinguish between objects 
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
        
        self.digital_map = np.copy(self.M_KB)
        self.state = np.ones(self.grid_size,dtype=np.int32)
        robot_pos = np.argwhere(self.M_KB == self.robot_code)[0]
        goal_pos = np.argwhere(self.M_KB == self.goal_code)[0]
        self.state[tuple(robot_pos)] = self.robot_code
        self.state[tuple(goal_pos)] = self.goal_code
        
        self.setup_environment()
        self.robot_pos = self.get_robot_position()
        
        return self.state
            
    def get_fov_info(self, pos):
        if 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]:
            return self.digital_map[tuple(pos)] 
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
        
        self.digital_map[tuple(old_pos)] = self.free_space
        self.digital_map[tuple(next_pos)] = self.robot_code
    
    def update_robot_movement_state(self, old_pos, new_pos, cell_code):
        logging.info(f'-| Updating Robot Movement State : {old_pos} -> {new_pos} |  FOV CODE : {cell_code} |-')
        
        if tuple(old_pos) != tuple(new_pos):
            self.robot_movement_state[tuple(new_pos)] += 1
        
        if cell_code in [self.wall_code,self.room_code, self.non_movable_code]:
            self.robot_movement_state[tuple(new_pos)] *= 2  # Scale higher visit for objects
    
    def translate_action(self,action,current_pos,next_pos,next_cell_code):
        if next_cell_code not in [self.wall_code, self.non_movable_code, self.room_code,self.movable_code,self.goal_code]: # If Free Space
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
            if next_cell_code == self.non_movable_code:
                self.state[tuple(next_pos)] = self.object_code # updates observation
            else:
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
            if new_obj_pos_fov_cell_code not in [self.non_movable_code,self.wall_code,self.room_code,self.movable_code,self.goal_code]:
                self.update_robot_position_in_state(current_pos,next_pos)
                self.update_robot_movement_state(current_pos,next_pos,next_cell_code)
                self.state[tuple(new_obj_pos)] = self.object_code
                self.digital_map[tuple(new_obj_pos)] = self.movable_code
                action_info = {
                    "object_type":next_cell_code,
                    "movablity":True,
                    "has_reached_goal":False,
                    "is_freespace":False
                }
            else:
                self.update_robot_position_in_state(current_pos,current_pos)
                self.update_robot_movement_state(current_pos,current_pos,self.non_movable_code) # manually updating visit for this state to be higher so it doesnt visit it
                self.state[tuple(next_pos)] = self.object_code # updates observation
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
    
    def calculate_max_distance(self):
        # Calculate distances from the goal to each of the four corners
        goal_pos = self.get_goal_position()
        distances = [
            math.sqrt((goal_pos[0] - 0) ** 2 + (goal_pos[1] - 0) ** 2),           # Distance to bottom-left corner
            math.sqrt((goal_pos[0] - self.grid_size[0]) ** 2 + (goal_pos[1] - 0) ** 2),       # Distance to bottom-right corner
            math.sqrt((goal_pos[0] - 0) ** 2 + (goal_pos[1] - self.grid_size[1]) ** 2),      # Distance to top-left corner
            math.sqrt((goal_pos[0] - self.grid_size[0]) ** 2 + (goal_pos[1] - self.grid_size[1]) ** 2)   # Distance to top-right corner
        ]
        return max(distances)
    
    def calculate_reward(self,next_pos,next_cell_code) :
        reward = 0
        done = False
        goal_reached = False
        
        if next_cell_code in [self.wall_code,self.room_code]: # penalize for collision with wall and room
            reward -= self.env_config['wall_penalty']
            # done = True
        
        if self.robot_movement_state[tuple(next_pos)] == 1: # reward for new visits - promotes exploration
            reward += self.env_config['exploration_reward']
        
        if next_cell_code == self.goal_code:  # reward for reaching goal
            base_reward = self.env_config['goal_base_reward']
            remaining_steps_ratio = (self.max_steps - self.current_step) / self.max_steps
            additional_reward = remaining_steps_ratio *  self.env_config['goal_base_reward']
            reward += min(base_reward + additional_reward,  self.env_config['goal_base_reward'] *  self.env_config['goal_factor'])
            logging.info(f"[GOAL] Goal Reached @ {self.episode_count}")
            goal_reached = True
            done = True
        
        # Distance reward to goal - scale from 0 to 5
        if self.env_config['distance_reward']:
            goal_pos = self.get_goal_position()
            max_rel_goal_dist = self.calculate_max_distance()
            distance = math.sqrt((next_pos[0] - goal_pos[0]) ** 2 + (next_pos[0] - goal_pos[1]) ** 2)
            normalized_distance = distance / max_rel_goal_dist
            reward += (1 - normalized_distance) * 5
        
        if self.current_step >= self.max_steps:
            done = True
        
        return reward,done,goal_reached
    
    def log_interactions(self): # will log interactions per episode
        wandb.log({"immovable_interactions":self.cumulative_immovable_interactions})
        wandb.log({"movable_interactions":self.cumulative_movable_interactions})
            
    def step(self, action):
        done = False
        self.robot_pos = self.get_robot_position()
        next_pos = self.get_fov_cell(self.robot_pos, action)
        cell_code = self.get_fov_info(next_pos)
        reward,done,goal_reached = self.calculate_reward(next_pos,cell_code)
        self.translate_action(action,self.robot_pos,next_pos,cell_code)
        self.cumulative_reward += reward        
        wandb.log({"episode":self.episode_count,"step":self.current_step})
        wandb.log({"step":self.current_step,"reward":reward})
        info = {
            "goal_reached":goal_reached,
            "episode_count": self.episode_count,
            "current_step":self.current_step,
            "cumulative_reward":self.cumulative_reward,
            "cumulative_interactions":sum([self.cumulative_movable_interactions,self.cumulative_immovable_interactions]),
            "movable_interactions":self.cumulative_movable_interactions,
            "non_movable_interactions":self.cumulative_immovable_interactions,
            "goal_reward":reward
        }
        logging.info(f"-EPISODE:{self.episode_count} @ STEP:{self.current_step}- Reward : {reward}")
        if done:
            self.log_interactions()
            wandb.log({"episode":self.episode_count,"cummulative_reward":self.cumulative_reward})
            self.episode_count += 1
        self.current_step += 1
        return self.state, reward, done, info

    def normalize_array(self,array):
        if np.max(array) == np.min(array):
            return array
        else:
            return (array - np.min(array)) / (np.max(array) - np.min(array))

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
                fig, axs = plt.subplots(1,3,figsize=(40,10))
                for ax in axs:
                    ax.set_xlim(0, self.grid_size[0])
                    ax.set_ylim(0, self.grid_size[1])
                    ax.grid(True)
                for r in range(self.grid_size[0]):
                    for c in range(self.grid_size[1]):
                        color = self.colors[self.state[r, c]]
                        rect = Rectangle((c, (self.grid_size[0]-1)-r), 1, 1, color=color)
                        axs[0].add_patch(rect)
                        axs[0].set_title('Robot Observations Update')
                for r in range(self.grid_size[0]):
                    for c in range(self.grid_size[1]):
                        color = self.colors[self.digital_map[r, c]]
                        rect = Rectangle((c, (self.grid_size[0]-1)-r), 1, 1, color=color)
                        axs[1].add_patch(rect)
                        axs[1].set_title('Environment Digital Map')
                
                normalized_movement_state = self.normalize_array(self.robot_movement_state)
                sns.heatmap(normalized_movement_state, ax=axs[2], cmap='viridis', cbar=True)
                axs[2].set_title('Digital Map Visitation Heatmap')
                
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
                fig, axs = plt.subplots(1,3,figsize=(40,10))
                for ax in axs:
                    ax.set_xlim(0, self.grid_size[0])
                    ax.set_ylim(0, self.grid_size[1])
                    ax.grid(True)
                for r in range(self.grid_size[0]):
                    for c in range(self.grid_size[1]):
                        color = self.colors[self.state[r, c]]
                        rect = Rectangle((c, (self.grid_size[0]-1)-r), 1, 1, color=color)
                        axs[0].add_patch(rect)
                        axs[0].set_title('Robot Observations Update')
                for r in range(self.grid_size[0]):
                    for c in range(self.grid_size[1]):
                        color = self.colors[self.digital_map[r, c]]
                        rect = Rectangle((c, (self.grid_size[0]-1)-r), 1, 1, color=color)
                        axs[1].add_patch(rect)
                        axs[1].set_title('Environment Digital Map')
                        
                normalized_movement_state = self.normalize_array(self.robot_movement_state)
                sns.heatmap(normalized_movement_state, ax=axs[2], cmap='viridis', cbar=True)
                axs[2].set_title('Digital Map Visitation Heatmap')
                
                # Convert the Matplotlib figure to an RGB array
                canvas = agg.FigureCanvasAgg(fig)
                canvas.draw()
                width, height = fig.get_size_inches() * fig.get_dpi()
                image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
                
                plt.close(fig)  # Close the figure to prevent it from being displayed
                return image
