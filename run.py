import argparse
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from time import time
from functools import partial
from models import ADTOF_FrameRNN
from train import train

# Only run this file directly
assert __name__ == "__main__"

# Declare an argument parser for this file
parser = argparse.ArgumentParser("run.py")
parser.add_argument("device", help="The device to run experiments on", type=str, default="cuda")

# ----------------------------------------------------------------------------------------------------------------

num_samples = 5

train_path = "data/adtof/adtof_yt_train"
val_path = "data/adtof/adtof_yt_validation"

config = {
    "batch_size": tune.choice([16, 32, 64]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "weight_decay": tune.loguniform(1e-2, 1e-4),
    "amsgrad": tune.choice([True, False]),
}

Model = ADTOF_FrameRNN

device = parser.device
seed = time()

scheduler = ASHAScheduler(
    metric="loss",
    mode="min"
)

# Run the experiments
result = tune.run(
    partial(train, Model=Model, train_path=train_path, val_path=val_path, device=device, seed=seed),
    config=config,
    num_samples=num_samples,
    scheduler=scheduler
)

# Print the results
best_trial = result.get_best_trial()
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final validation loss: {best_trial.last_result["loss"]}")