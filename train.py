import gym_env.custom_env 
import gym_env.search_and_rescue 
import gym_env.search_and_rescue_brainless
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
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
import argparse
import pdb
from contextlib import contextmanager

@contextmanager
def prefixed_wandb_log(prefix):
    original_log = wandb.log
    
    # Override wandb.log to prepend the prefix
    def modified_log(data, *args, **kwargs):
        prefixed_data = {f"{prefix}/{k}": v for k, v in data.items()}
        original_log(prefixed_data, *args, **kwargs)
    
    wandb.log = modified_log
    try:
        yield
    finally:
        wandb.log = original_log

parser = argparse.ArgumentParser()
parser.add_argument('config_path',type=str,help='Config Path',default='config.json')
args = parser.parse_args()

CONFIG = read_config(args.config_path)
log_dir = check_and_create_directory(os.path.join(os.getcwd(),CONFIG['log_dir']))

def always_record(episode_id):
    return True  # This will ensure every episode is recorded

def make_env():
    env = gym.make(CONFIG["environment"]["name"], render_mode="rgb_array")
    # env = gym.wrappers.RecordVideo(env, f"{log_dir}/videos",episode_trigger=always_record)  # record videos
    # env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    return env

def log_to_csv(cumulative_data,csv_file_path):
    columns=["goal_reached", "episode_count", "current_step", "cumulative_reward", "cumulative_interactions", "movable_interactions", "non_movable_interactions", "goal_reward", "time_taken_for_episode"]
    if not os.path.exists(csv_file_path):
        mode = 'w' 
    else:
        mode = 'a'  
    new_data_df = pd.DataFrame([cumulative_data], columns=columns)
    
    if mode == 'w':
        new_data_df.to_csv(csv_file_path, mode=mode, index=False)
    else:
        new_data_df.to_csv(csv_file_path, mode=mode, index=False, header=False)

def log_to_wandb(csv_file_path):
    columns = ["goal_reached", "episode_count", "current_step", "cumulative_reward", "cumulative_interactions", "movable_interactions", "non_movable_interactions", "goal_reward", "time_taken"]
    df = pd.read_csv(csv_file_path)
    results_table = wandb.Table(columns=columns)
    for index, row in df.iterrows():
        results_table.add_data(*row)
    wandb.log({"Train/results_table": results_table})

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

env = DummyVecEnv([make_env])
model = A2C(config['policy'], env, verbose=1) 

# model.learn(total_timesteps=config['timesteps'], callback=callback) # No Control over model save

# Custom Training Loop
total_timesteps = config['timesteps'] 
episode_rewards = []  
best_mean_reward = -float('inf')  
best_episode = 0 
obs = env.reset()  
episode_count = 0  
sum_rewards = 0  
start_timer = time.time()
cumulative_data = []
csv_file_path = os.path.join(log_dir,'results_summary.csv')
eval_interval = total_timesteps // CONFIG['eval_interval']

for step in range(total_timesteps):
    with prefixed_wandb_log("Train"):
        logging.info(f'- Training: {step}/{total_timesteps} @ episode {episode_count} -')
        action, _ = model.predict(obs,deterministic=False)
        obs, reward, done, info = env.step(action)
        if done:
            time_taken_for_episode = time.time() - start_timer
            episode_rewards.append(sum_rewards)  
            if info[0]['goal_reached']:
                cumulative_data = [
                    info[0]["goal_reached"],
                    info[0]["episode_count"],
                    info[0]["current_step"],
                    info[0]["cumulative_reward"],
                    info[0]["cumulative_interactions"],
                    info[0]["movable_interactions"],
                    info[0]["non_movable_interactions"],
                    info[0]["goal_reward"],
                    time_taken_for_episode
                ]
                log_to_csv(cumulative_data,csv_file_path)
            episode_count += 1  
            # sum_rewards = 0  
            obs = env.reset()  
            start_timer = time.time()
    if (step + 1) % eval_interval == 0 or step == total_timesteps - 1:
        with prefixed_wandb_log("Eval"):
            mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=CONFIG['eval_episodes'],deterministic=False)
            logging.info(f"Evaluation at step {step+1}: mean reward = {mean_reward}, std. dev. = {std_reward}")
            wandb.log({"mean_reward":mean_reward,"eval_step":step+1})
            wandb.log({"std_reward":std_reward,"eval_step":step+1})
            model.save(os.path.join(log_dir,f"final_model_{step+1}.zip"))
    
    
        
log_to_wandb(csv_file_path)
