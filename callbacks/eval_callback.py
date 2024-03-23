from stable_baselines3.common.evaluation import evaluate_policy
import wandb
import os 
from stable_baselines3.common.callbacks import BaseCallback
import logging

class TrainingAndEvaluationCallback(BaseCallback):
    def __init__(self, model, env, n_eval_episodes=10, eval_freq=1000, log_dir='./', verbose=1):
        super().__init__(verbose)
        self.model = model
        self.env = env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -float('inf')
        self.log_dir = log_dir

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            mean_reward, _ = evaluate_policy(self.model, self.env, n_eval_episodes=self.n_eval_episodes)
            wandb.log({"mean_reward": mean_reward})

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                logging.info(f"[INFO] Saving model for Mean Reward : {mean_reward}")
                self.model.save(os.path.join(self.log_dir, "best_model"))
        
        return True  # Return True to continue training

    def _on_training_end(self):
        # Perform prediction with the trained model, for example:
        obs = self.env.reset()
        for _ in range(1000):  # Run 1000 steps for demonstration
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _ = self.env.step(action)
            if done:
                obs = self.env.reset()

        # Save the videos to WandB
        if done:
            self.model.save(os.path.join(self.log_dir, "final_model"))
            
            # for video_file in os.listdir(os.path.join(os.getcwd(), "videos")):
            #     if video_file.endswith('.mp4'):
            #         video_filepath = os.path.join(os.getcwd(), "videos", video_file)
            #         episode_id = video_file.split('.')[0].split('-')[-1]
            #         wandb.log({f"video-{episode_id}": wandb.Video(video_filepath)})
                
        wandb.finish()