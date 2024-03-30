from contextlib import contextmanager
import wandb

@contextmanager
def prefixed_wandb_log(prefix):
    original_log = wandb.log
    
    # Override wandb.log to prepend the prefix
    def modified_log(data, *args, **kwargs):
        prefixed_data = {f"{prefix}/{k}": v for k, v in data.items()}
        original_log(prefixed_data, *args, **kwargs)
    
    wandb.log = modified_log
    try:
        yield
    finally:
        wandb.log = original_log