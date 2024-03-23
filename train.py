import gym_env.custom_env 
import gym_env.search_and_rescue 
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
import gym 
import os 
import logging
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
from utils.helper import read_config
from callbacks.eval_callback import TrainingAndEvaluationCallback
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('config_path',type=str,help='Config Path',default='config.json')
args = parser.parse_args()

CONFIG = read_config(args.config_path)

def always_record(episode_id):
    return True  # This will ensure every episode is recorded

def make_env():
    env = gym.make(CONFIG["environment"], render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, f"videos",episode_trigger=always_record)  # record videos
    env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    return env

config = {
    "model": CONFIG['algorithm'],
    "policy":CONFIG['policy'],
    "timesteps":CONFIG['total_timesteps'],
    "env":CONFIG['environment']
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
callback = TrainingAndEvaluationCallback(model, env, n_eval_episodes=10, eval_freq=1000, log_dir='logs')
model.learn(total_timesteps=config['timesteps'], callback=callback)