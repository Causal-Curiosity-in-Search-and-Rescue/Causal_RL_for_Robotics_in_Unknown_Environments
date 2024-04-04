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
from utils.log_helper import calculate_aggregate_stats
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
n_envs = 1
n_eval_episodes = CONFIG["eval_episodes"]

episode_rewards = []
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
                
    observations = new_observations
    
mean_goal_reqched , std_goal_reached  = calculate_aggregate_stats(collected_dictionary["goal_reached"])
logging.info(f"Mean goal Reached : {mean_goal_reqched} | Std. Goal reached : {std_goal_reached}")
