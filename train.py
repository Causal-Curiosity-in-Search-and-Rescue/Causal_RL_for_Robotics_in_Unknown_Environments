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
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
from utils.helper import read_config,check_and_create_directory
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('config_path',type=str,help='Config Path',default='config.json')
args = parser.parse_args()

CONFIG = read_config(args.config_path)
log_dir = check_and_create_directory(os.path.join(os.getcwd(),CONFIG['log_dir']))

def always_record(episode_id):
    return True  # This will ensure every episode is recorded

def make_env():
    env = gym.make(CONFIG["environment"]["name"], render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, f"{log_dir}/videos",episode_trigger=always_record)  # record videos
    env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    return env

def log_to_wandb(goal_reached,episode_count, current_step, cumulative_reward, cumulative_interactions,movable_interactions,non_movable_interactions, goal_reward,time_taken):
    if 'summary_table' not in wandb.run.summary:
        columns = ["goal_reached", "episode_count", "current_step","cumulative_reward", "cumulative_interactions", "movable_interactions","non_movable_interactions","goal_reward","time_taken"]
        wandb.run.summary['results_table'] = wandb.Table(columns=columns)
   
    wandb.run.summary['results_table'].add_data(goal_reached,episode_count, current_step, cumulative_reward, cumulative_interactions,movable_interactions,non_movable_interactions, goal_reward,time_taken)
    wandb.log({
        "goal_reached":goal_reached,
        "episode_count": episode_count,
        "current_step":current_step,
        "cumulative_reward":cumulative_reward,
        "cumulative_interactions":cumulative_interactions,
        "movable_interactions":movable_interactions,
        "non_movable_interactions":non_movable_interactions,
        "goal_reward":reward,
        "time_taken":time_taken
    })

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

for step in range(total_timesteps):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        time_taken_for_episode = time.time() - start_timer
        episode_rewards.append(sum_rewards)  
        model.save(os.path.join(log_dir, f'final_model_{episode_count}.zip')) 
        if step >= CONFIG['environment']['max_steps'] - 1:
            log_to_wandb(
                info[0]["goal_reached"],
                info[0]["episode_count"],
                info[0]["current_step"],
                info[0]["cumulative_reward"],
                info[0]["cumulative_interactions"],
                info[0]["movable_interactions"],
                info[0]["non_movable_interactions"],
                info[0]["goal_reward"],
                time_taken_for_episode
            )
        elif info['goal_reached']:
            log_to_wandb(
                info[0]["goal_reached"],
                info[0]["episode_count"],
                info[0]["current_step"],
                info[0]["cumulative_reward"],
                info[0]["cumulative_interactions"],
                info[0]["movable_interactions"],
                info[0]["non_movable_interactions"],
                info[0]["goal_reward"],
                time_taken_for_episode
            )
        episode_count += 1  
        # sum_rewards = 0  
        obs = env.reset()  
        start_timer = time.time()
        

logging.info(f"Total episodes: {episode_count}")
logging.info(f"Average reward: {np.mean(episode_rewards)}")
logging.info(f"Best episode: {best_episode} with reward: {best_mean_reward}")