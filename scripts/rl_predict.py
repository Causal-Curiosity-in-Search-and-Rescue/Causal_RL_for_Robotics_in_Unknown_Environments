import argparse
from utils.helper import read_config
from stable_baselines3 import PPO,A2C
import gym_env.custom_env 
import gym 
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

parser = argparse.ArgumentParser()
parser.add_argument('config_path',type=str,help='Config Path',default='config.json')
args = parser.parse_args()

CONFIG = read_config(args.config_path)

def make_env():
    env = gym.make("CustomEnv-v0", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, f"videos")  # record videos
    env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    return env

env = DummyVecEnv([make_env])
    
if CONFIG['rl_config']['algorithm'] == 'A2C':
    model = A2C.load("a2c_crl")
else: 
    model = PPO.load("ppo_crl")

obs = env.reset()
initial_action = 0
set_initial_action  = True
while True:
    if set_initial_action:
        action = initial_action
        set_initial_action=False
    else:
        action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render(mode='human')
    if dones:
        obs = env.reset()