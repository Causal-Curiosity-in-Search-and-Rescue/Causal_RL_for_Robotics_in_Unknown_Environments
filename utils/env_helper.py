import gym_env.custom_env 
import gym_env.search_and_rescue 
import gym_env.search_and_rescue_brainless
import gym 
import numpy as np

def always_record(episode_id):
    return True  # This will ensure every episode is recorded

def make_env(env_id):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        #video_id = np.random.randint(0,100)
        #env = gym.wrappers.RecordVideo(env, f"videos",episode_trigger=always_record)  # record videos
        # env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
        return env
    return _init