from utils.env_helper import make_env,make_env_for_inference
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
import numpy as np 
import time 
from utils.log_helper import log_aggregate_stats,log_to_csv,log_results_table_to_wandb,calculate_aggregate_stats
import os
import pdb

def evaluate(model,CONFIG,step,eval_csv_file_path):
    n_envs = CONFIG["environment"]["num_parallel_env"]
    env_ids = [CONFIG["environment"]["name"]] * n_envs
    envs = [make_env(env_id) for env_id in env_ids]
    env = SubprocVecEnv(envs)
    
    n_eval_episodes = CONFIG["eval_episodes"]
    
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
                    
                    if info['goal_reached']:
                        cumulative_data = [
                            i,
                            info["goal_reached"],
                            info["episode_count"],
                            info["current_step"],
                            info["cumulative_reward"],
                            info["cumulative_interactions"],
                            info["movable_interactions"],
                            info["non_movable_interactions"],
                            info["goal_reward"],
                            time_taken_for_episode
                        ]
                        log_to_csv(cumulative_data,eval_csv_file_path)
                    
        observations = new_observations
        
    log_aggregate_stats(collected_dictionary,key="cumulative_reward",log_string="cumulative_reward",step=step)
    log_aggregate_stats(collected_dictionary,key="cumulative_interactions",log_string="cumulative_interactions",step=step)
    log_aggregate_stats(collected_dictionary,key="movable_interactions",log_string="movable_interactions",step=step)
    log_aggregate_stats(collected_dictionary,key="non_movable_interactions",log_string="non_movable_interactions",step=step)
    log_aggregate_stats(collected_dictionary,key="goal_reward",log_string="goal_reward",step=step)
    log_aggregate_stats(collected_dictionary,key="goal_reached",log_string="goal_reached",step=step)
    log_aggregate_stats(collected_dictionary,key="time_taken_per_episode",log_string="time_taken_per_episode",step=step)

    mean_goal_rchd,std_reached = calculate_aggregate_stats(collected_dictionary["goal_reached"])

    if mean_goal_rchd == 1 and std_reached == 0:
        return 1
    # elif mean_goal_rchd == 0 and std_reached == 0:
    #     return 1
    else:
        return 0


def inference(model,CONFIG):
    env = make_env_for_inference(CONFIG["environment"]["name"],render_video=True)
    pdb.set_trace()
    env = DummyVecEnv([lambda: env])
    n_envs = 1
    n_eval_episodes = CONFIG["eval_episodes"]
    
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
        
    return calculate_aggregate_stats(collected_dictionary["goal_reached"])
