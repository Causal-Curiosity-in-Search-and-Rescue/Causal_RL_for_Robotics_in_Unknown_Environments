import gym_env.custom_env 
import gym_env.search_and_rescue 
import gym_env.search_and_rescue_brainless
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
import time
import wandb
import numpy as np 
import gym 
import os 
import logging
import pandas as pd 
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from utils.helper import read_config,check_and_create_directory
from utils.log_helper import log_to_csv,log_results_table_to_wandb,log_aggregate_stats
from utils.wandb_context import prefixed_wandb_log
import argparse
import pdb

def always_record(episode_id):
    return True  # This will ensure every episode is recorded

def make_env(env_id):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        # env = gym.wrappers.RecordVideo(env, f"{log_dir}/videos",episode_trigger=always_record)  # record videos
        # env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
        return env
    return _init

def evaluate(model,CONFIG,step,eval_csv_file_path):
    n_envs = CONFIG["environment"]["num_parallel_env"]
    env_ids = [CONFIG["environment"]["name"]] * n_envs
    envs = [make_env(env_id) for env_id in env_ids]
    env = SubprocVecEnv(envs)
    
    n_eval_episodes = CONFIG["eval_episodes"]
    
    episode_counts = np.zeros(n_envs, dtype="int")
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")
    
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    start_times = np.array([time.time()] * n_envs)  # Start times for each env
   
    collected_dictionary = {
        "cumulative_reward":[],
        "cumulative_interactions":[],
        "movable_interactions":[],
        "non_movable_interactions":[],
        "goal_reward":[],
        "goal_reached":[],
        "time_taken_per_episode":[]
    }
    
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  
            state=states,
            episode_start=episode_starts,
            deterministic=False,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done
                if done:
                    episode_counts[i] += 1
                    time_taken_for_episode = time.time() - start_times[i]
                    start_times[i] = time.time()  # Reset start time for the next episode
                    
                    collected_dictionary["cumulative_reward"].append(info["cumulative_reward"])
                    collected_dictionary["cumulative_interactions"].append(info["cumulative_interactions"])
                    collected_dictionary["movable_interactions"].append(info["movable_interactions"])
                    collected_dictionary["non_movable_interactions"].append(info["non_movable_interactions"])
                    collected_dictionary["goal_reward"].append(info["goal_reward"])
                    collected_dictionary["goal_reached"].append(info["goal_reached"])
                    collected_dictionary["time_taken_per_episode"].append(time_taken_for_episode)
                    
                    if info['goal_reached']:
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
                            time_taken_for_episode
                        ]
                        log_to_csv(cumulative_data,eval_csv_file_path)
                    
        observations = new_observations
        
    log_aggregate_stats(collected_dictionary,key="cumulative_reward",log_string="cumulative_reward",step=step)
    log_aggregate_stats(collected_dictionary,key="cumulative_interactions",log_string="cumulative_interactions",step=step)
    log_aggregate_stats(collected_dictionary,key="movable_interactions",log_string="movable_interactions",step=step)
    log_aggregate_stats(collected_dictionary,key="non_movable_interactions",log_string="non_movable_interactions",step=step)
    log_aggregate_stats(collected_dictionary,key="goal_reward",log_string="goal_reward",step=step)
    log_aggregate_stats(collected_dictionary,key="goal_reached",log_string="goal_reached",step=step)
    log_aggregate_stats(collected_dictionary,key="time_taken_per_episode",log_string="time_taken_per_episode",step=step)

def train_and_evaluate(CONFIG):
    num_envs = CONFIG["environment"]["num_parallel_env"]
    env_ids = [CONFIG["environment"]["name"]] * num_envs
    envs = [make_env(env_id) for env_id in env_ids]
    env = SubprocVecEnv(envs)
   
    model = A2C(CONFIG['policy'], env, verbose=1) 

    total_timesteps = CONFIG['total_timesteps'] 
    obs = env.reset()  
    episode_counts = np.zeros(num_envs, dtype=int)  # Tracking episodes for each env
    start_times = np.array([time.time()] * num_envs)  # Start times for each env
    eval_interval = total_timesteps // CONFIG['eval_interval']
    log_interval = total_timesteps // CONFIG['log_interval']

    cumulative_data = []
    train_csv_file_path = os.path.join(log_dir,'train_results_summary.csv')
    eval_csv_file_path = os.path.join(log_dir,'eval_results_summary.csv')
    
    collected_dictionary = {
        "cumulative_reward":[],
        "cumulative_interactions":[],
        "movable_interactions":[],
        "non_movable_interactions":[],
        "goal_reward":[],
        "goal_reached":[],
        "time_taken_per_episode":[]
    }
    
    for step in range(total_timesteps):
        
        # Training Mode
        with prefixed_wandb_log("Train"):
            logging.info(f'- Training: {step}/{total_timesteps} -')
            actions, _ = model.predict(obs, deterministic=False)
            obs, rewards, dones, infos = env.step(actions)
            
            if (step + 1) % log_interval == 0 or step == total_timesteps - 1:
                log_aggregate_stats(collected_dictionary,key="cumulative_reward",log_string="cumulative_reward",step=step+1)
                log_aggregate_stats(collected_dictionary,key="cumulative_interactions",log_string="cumulative_interactions",step=step+1)
                log_aggregate_stats(collected_dictionary,key="movable_interactions",log_string="movable_interactions",step=step+1)
                log_aggregate_stats(collected_dictionary,key="non_movable_interactions",log_string="non_movable_interactions",step=step+1)
                log_aggregate_stats(collected_dictionary,key="goal_reward",log_string="goal_reward",step=step+1)
                log_aggregate_stats(collected_dictionary,key="goal_reached",log_string="goal_reached",step=step+1)
                log_aggregate_stats(collected_dictionary,key="time_taken_per_episode",log_string="time_taken_per_episode",step=step+1)
                collected_dictionary = {
                    "cumulative_reward":[],
                    "cumulative_interactions":[],
                    "movable_interactions":[],
                    "non_movable_interactions":[],
                    "goal_reward":[],
                    "goal_reached":[],
                    "time_taken_per_episode":[]
                }
                
            for i, done in enumerate(dones):
                if done:
                    episode_counts[i] += 1
                    info = infos[i]  # Get info for the done environment
                    time_taken_for_episode = time.time() - start_times[i]
                    start_times[i] = time.time()  # Reset start time for the next episode
                    
                    collected_dictionary["cumulative_reward"].append(info["cumulative_reward"])
                    collected_dictionary["cumulative_interactions"].append(info["cumulative_interactions"])
                    collected_dictionary["movable_interactions"].append(info["movable_interactions"])
                    collected_dictionary["non_movable_interactions"].append(info["non_movable_interactions"])
                    collected_dictionary["goal_reward"].append(info["goal_reward"])
                    collected_dictionary["goal_reached"].append(info["goal_reached"])
                    collected_dictionary["time_taken_per_episode"].append(time_taken_for_episode)
                    
                    if info[i]['goal_reached']:
                        cumulative_data = [
                            i,
                            info[i]["goal_reached"],
                            info[i]["episode_count"],
                            info[i]["current_step"],
                            info[i]["cumulative_reward"],
                            info[i]["cumulative_interactions"],
                            info[i]["movable_interactions"],
                            info[i]["non_movable_interactions"],
                            info[i]["goal_reward"],
                            time_taken_for_episode
                        ]
                        log_to_csv(cumulative_data,train_csv_file_path)
                    
                    obs[i] = env.reset(index=i)
        
        # Evaluation Mode
        if (step + 1) % eval_interval == 0 or step == total_timesteps - 1:
            with prefixed_wandb_log("Eval"):
                evaluate(model,CONFIG,step+1,eval_csv_file_path)
                model.save(os.path.join(log_dir,f"final_model_{step+1}.zip"))
        
    log_results_table_to_wandb(train_csv_file_path,prefix='Train')
    log_results_table_to_wandb(eval_csv_file_path,prefix='Eval')


def main(CONFIG):
    train_and_evaluate(CONFIG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path',type=str,help='Config Path',default='config.json')
    args = parser.parse_args()

    CONFIG = read_config(args.config_path)
    log_dir = check_and_create_directory(os.path.join(os.getcwd(),CONFIG['log_dir']))
    
    config = {
        "model": CONFIG['algorithm'],
        "policy":CONFIG['policy'],
        "timesteps":CONFIG['total_timesteps'],
        "env":CONFIG["environment"]["name"]
    }
    wandb.init(
        config=config,
        entity=CONFIG['wandb']['entity'],
        project=CONFIG['wandb']['project'],
        monitor_gym=True,       # automatically upload gym environements' videos
        save_code=True,
    )
    
    main(CONFIG)
