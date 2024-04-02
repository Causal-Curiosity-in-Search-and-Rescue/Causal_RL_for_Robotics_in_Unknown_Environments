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
from callbacks.train_callback import MetricsCallback
from utils.env_helper import make_env
import argparse
import pdb


def train_and_evaluate(CONFIG):
    num_envs = CONFIG["environment"]["num_parallel_env"]
    env_ids = [CONFIG["environment"]["name"]] * num_envs
    envs = [make_env(env_id) for env_id in env_ids]
    env = SubprocVecEnv(envs)
   
    model = A2C(CONFIG['policy'], env, verbose=1) 

    total_timesteps = CONFIG['total_timesteps'] 
    model.set_env(env=env)
    callbacks = MetricsCallback()
    # Training Mode
    model.learn(total_timesteps=total_timesteps,reset_num_timesteps=False,callback=callbacks)


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
