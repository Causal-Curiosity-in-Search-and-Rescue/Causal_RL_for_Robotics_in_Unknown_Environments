import gym_env.custom_env 
import gym_env.search_and_rescue 
import gym_env.search_and_rescue_brainless
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
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

model = A2C.load(CONFIG["inference"]["model_path"])

env = DummyVecEnv([make_env])
obs = env.reset()

total_episodes = CONFIG["inference"]["num_episodes"]  
episode_rewards = []

for episode in range(total_episodes):
    done = False
    sum_rewards = 0  

    while not done:
        action, _states = model.predict(obs, deterministic=True)  # deterministic=True for reproducible actions
        obs, reward, done, info = env.step(action)
        sum_rewards += reward

    episode_rewards.append(sum_rewards)  
    obs = env.reset()  

average_reward = np.mean(episode_rewards)
logging.info(f"Completed {total_episodes} episodes with an average reward of {average_reward}")