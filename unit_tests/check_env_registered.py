import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import gym 
import gym_env.custom_env
import gym_env.search_and_rescue 
import gym_env.search_and_rescue_brainless
import logging
import wandb
import pdb

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# wandb.init(
#     project="Testing"
# )

try:
    env = gym.make('SearchAndRescueNoCausalEnv-v0')
    observation = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()  # Random action
        observation, reward, done, info = env.step(action)
        if info["episode_ended"]:
            observation = env.reset()
    logger.info("Test successful: Custom environment initialized and interacted with successfully.")
except Exception as e:
    logger.error(f"Test failed: {e}")
