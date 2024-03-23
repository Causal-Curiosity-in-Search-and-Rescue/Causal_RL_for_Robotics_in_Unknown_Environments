import optuna
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import gym_env.custom_env 
import gym_env.search_and_rescue 
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb
import gym 
from callbacks.wandb_callback import WandBCallback
import argparse
from utils.helper import read_config

parser = argparse.ArgumentParser()
parser.add_argument('config_path',type=str,help='Config Path',default='config.json')
args = parser.parse_args()

CONFIG = read_config(args.config_path)

def make_env():
    env = gym.make("CustomEnv-v0", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, f"videos")  # record videos
    env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    return env

def objective(trial):
    run = wandb.init(project=CONFIG['wandb']['project'], entity=CONFIG['wandb']['username'], reinit=True)
    
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.0001, 0.1)
    
    
    wandb.init(
        config=CONFIG,
        sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
        project="rl_learning",
        monitor_gym=True,       # automatically upload gym environements' videos
        save_code=True,
    )
    
    wandb.config.update({
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
    })
    
    env = DummyVecEnv([make_env])
    
    if CONFIG['rl_config']['algorithm'] == 'A2C':
        model = A2C("MlpPolicy", env, learning_rate=learning_rate, ent_coef=ent_coef,verbose=1, tensorboard_log=f"runs/a2c")
    else: 
        model = PPO("MlpPolicy", env, learning_rate=learning_rate, ent_coef=ent_coef,verbose=1, tensorboard_log=f"runs/ppo")
    
    model.learn(total_timesteps=CONFIG['rl_config']['total_timesteps'], callback=WandBCallback())
    
    mean_reward, _ = evaluate_policy(model, env, n_eval_emean_rewardpisodes=10)
    wandb.log({"mean_reward": mean_reward})
    model.save(f"a2c_crl_{trial}_{mean_reward}")
    run.finish()
    
    return mean_reward


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
