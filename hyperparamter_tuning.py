import optuna
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import gym_env.custom_env 
import gym_env.search_and_rescue 
from stable_baselines3 import PPO,A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
import wandb
import gym 
#from callbacks.wandb_callback import WandBCallback
import argparse
from utils.helper import read_config
from callbacks.train_callback import MetricsCallback


parser = argparse.ArgumentParser()
parser.add_argument('config_path',type=str,help='Config Path',default='config.json')
args = parser.parse_args()

CONFIG = read_config(args.config_path)

def make_env():
    env = gym.make("CustomEnv-v0", render_mode="human")
    #env = gym.wrappers.RecordVideo(env, f"videos")  # record videos
    env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
    return env

def objective(trial):
    wandb.init(
        config=CONFIG,
        entity=CONFIG['wandb']['entity'],
        project=CONFIG['wandb']['project'],
        monitor_gym=True,       # automatically upload gym environements' videos
        save_code=True,
    )
    # Hyperparameters to be tuned by Optuna
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    gamma = trial.suggest_uniform('gamma', 0.8, 0.9999)
    n_steps = trial.suggest_int('n_steps', 1, 2048)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00001, 0.1)
    vf_coef = trial.suggest_uniform('vf_coef', 0.5, 1)
    max_grad_norm = trial.suggest_uniform('max_grad_norm', 0.5, 5)
    
    # Assuming CONFIG is globally accessible or passed as an argument to the objective function
    #num_envs = 1
    #env_id = CONFIG["environment"]["name"]
    env = make_env()
    env = DummyVecEnv([lambda: env])
    
    env.reset()

    optimizer_eps = 1e-5  # Example, adjust according to your needs
    
    model = A2C(
        CONFIG['policy'], 
        env, 
        verbose=1,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs={"optimizer_kwargs":{"eps":optimizer_eps}}
    )


    callbacks = MetricsCallback()
    model.learn(total_timesteps=CONFIG['total_timesteps'], reset_num_timesteps=False, callback=callbacks)
    
    mean_goal_reached = callbacks.mean_goal_reached

    return mean_goal_reached


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

best_params = study.best_params
print('Best trial:', best_params)

wandb.log({"Best Parameters": best_params})
wandb.finish()