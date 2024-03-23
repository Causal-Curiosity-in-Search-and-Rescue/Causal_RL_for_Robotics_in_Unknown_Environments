from gym.envs.registration import register
from .custom_env import CustomEnv

register(
    id='CustomEnv-v0',
    entry_point='gym_env.custom_env:CustomEnv',
)

register(
    id='SearchAndRescueEnv-v0',
    entry_point='gym_env.search_and_rescue:SearchAndRescueEnv',
)

