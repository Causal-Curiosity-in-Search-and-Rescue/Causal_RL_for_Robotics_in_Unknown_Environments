import gym_env.custom_env 
import gym_env.search_and_rescue 
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import wandb
import gym 
import logging
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
from utils.helper import read_config
from callbacks.wandb_callback import WandBCallback
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('config_path',type=str,help='Config Path',default='config.json')
args = parser.parse_args()

CONFIG = read_config(args.config_path)

def make_env():
    env = gym.make(CONFIG["environment"], render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, f"videos")  # record videos
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
    entity="SR_GDP",
    project="[2D] Using Movability KB",
    monitor_gym=True,       # automatically upload gym environements' videos
    save_code=True,
)

env = DummyVecEnv([make_env])

model = A2C(config['policy'], env, verbose=1) 
model.learn(total_timesteps=config['timesteps'], callback=WandBCallback())
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
wandb.log({"mean_reward": mean_reward})
model.save(f"a2c_crl_{mean_reward}")

wandb.finish()