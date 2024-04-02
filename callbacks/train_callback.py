import wandb
from stable_baselines3.common.callbacks import BaseCallback
from utils.log_helper import log_aggregate_stats,log_to_csv,log_results_table_to_wandb
import os
from utils.wandb_context import prefixed_wandb_log
from utils.evaluate import evaluate
from utils.helper import read_config
import pdb
import numpy as np

class MetricsCallback(BaseCallback):
    
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_counts = None
        self.current_episode_rewards = None
        self.current_episode_lengths = None
        self.collected_dictionary = {
            "cumulative_reward":[],
            "cumulative_interactions":[],
            "movable_interactions":[],
            "non_movable_interactions":[],
            "goal_reward":[],
            "goal_reached":[],
            "time_taken_per_episode":[]
        }
        self.CONFIG = read_config()
        self.total_timesteps = self.CONFIG['total_timesteps']
        self.log_interval = self.total_timesteps // self.CONFIG['log_interval']
        self.eval_interval = self.total_timesteps // self.CONFIG['eval_interval']
        self.train_csv_file_path = os.path.join(os.getcwd(),self.CONFIG["log_dir"],'train_results_summary.csv')
        self.eval_csv_file_path = os.path.join(os.getcwd(),self.CONFIG["log_dir"],'eval_results_summary.csv')

    def _on_training_start(self):
        self.num_envs = self.model.env.num_envs
        self.episode_counts = [0] * self.num_envs
        self.current_episode_rewards = [0.0] * self.num_envs
        self.current_episode_lengths = [0] * self.num_envs
    
    def _on_step(self) -> bool:
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        infos = self.locals['infos']
        step = self.model.num_timesteps

        for i, info in enumerate(infos):
            if info["episode_ended"]:
                self.collected_dictionary["cumulative_reward"].append(info["cumulative_reward"])
                self.collected_dictionary["cumulative_interactions"].append(info["cumulative_interactions"])
                self.collected_dictionary["movable_interactions"].append(info["movable_interactions"])
                self.collected_dictionary["non_movable_interactions"].append(info["non_movable_interactions"])
                self.collected_dictionary["goal_reward"].append(info["goal_reward"])
                self.collected_dictionary["goal_reached"].append(info["goal_reached"])
                self.collected_dictionary["time_taken_per_episode"].append(info["time_taken_per_episode"])
            
            if info["goal_reached"]:
                cumulative_data = [
                    i,
                    info["goal_reached"],
                    info["episode_count"],
                    info["current_step"],
                    info["cumulative_reward"],
                    info["cumulative_interactions"],
                    info["movable_interactions"],
                    info["non_movable_interactions"],
                    info["goal_reward"],
                    info["time_taken_per_episode"]
                ]
                log_to_csv(cumulative_data,self.train_csv_file_path)

        for i in range(len(dones)):
            if np.sum(dones[i]) == self.num_envs: 
                self.model.env.reset()

        with prefixed_wandb_log("Train"):
            if (step + 1) % self.log_interval == 0 or step == self.total_timesteps - 1:
                log_aggregate_stats(self.collected_dictionary,key="cumulative_reward",log_string="cumulative_reward",step=step+1)
                log_aggregate_stats(self.collected_dictionary,key="cumulative_interactions",log_string="cumulative_interactions",step=step+1)
                log_aggregate_stats(self.collected_dictionary,key="movable_interactions",log_string="movable_interactions",step=step+1)
                log_aggregate_stats(self.collected_dictionary,key="non_movable_interactions",log_string="non_movable_interactions",step=step+1)
                log_aggregate_stats(self.collected_dictionary,key="goal_reward",log_string="goal_reward",step=step+1)
                log_aggregate_stats(self.collected_dictionary,key="goal_reached",log_string="goal_reached",step=step+1)
                log_aggregate_stats(self.collected_dictionary,key="time_taken_per_episode",log_string="time_taken_per_episode",step=step+1)
                self.collected_dictionary = {
                    "cumulative_reward":[],
                    "cumulative_interactions":[],
                    "movable_interactions":[],
                    "non_movable_interactions":[],
                    "goal_reward":[],
                    "goal_reached":[],
                    "time_taken_per_episode":[]
                }
        
        with prefixed_wandb_log("Eval"):
            if (step + 1) % self.eval_interval == 0 or step == self.total_timesteps - 1:
                evaluate(self.model,self.CONFIG,step+1,self.eval_csv_file_path)
                self.model.save(os.path.join(os.getcwd(),self.CONFIG["log_dir"],f"final_model_{step+1}.zip"))
        
        return True

    def _on_training_end(self):
        if os.path.exists(self.train_csv_file_path):
            log_results_table_to_wandb(self.train_csv_file_path,prefix='Train')
        if os.path.exists(self.eval_csv_file_path):
            log_results_table_to_wandb(self.eval_csv_file_path,prefix='Eval')