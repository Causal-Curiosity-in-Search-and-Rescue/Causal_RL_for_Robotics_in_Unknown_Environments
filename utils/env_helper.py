import gym_env.custom_env 
import gym_env.search_and_rescue 
import gym_env.search_and_rescue_brainless
import gym 

def always_record(episode_id):
    return True  # This will ensure every episode is recorded

def make_env(env_id):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        # env = gym.wrappers.RecordVideo(env, f"{log_dir}/videos",episode_trigger=always_record)  # record videos
        # env = gym.wrappers.RecordEpisodeStatistics(env)  # record stats such as returns
        return env
    return _init