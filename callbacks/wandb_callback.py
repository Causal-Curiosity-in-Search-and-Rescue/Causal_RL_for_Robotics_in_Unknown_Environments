import wandb
from stable_baselines3.common.callbacks import BaseCallback

class WandBCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(WandBCallback, self).__init__(verbose)
    
    def _on_step(self):
        wandb.log({"accumulated_reward": sum(self.locals["rewards"])})
        return True