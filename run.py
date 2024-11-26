import argparse
import torch
from ray import init, tune
from ray.tune.schedulers import ASHAScheduler
from time import time
from functools import partial
from models import ADTOF_FrameRNN
from train import train
from pathlib import Path

# Only run this file directly
assert __name__ == "__main__"

# Declare an argument parser for this file
parser = argparse.ArgumentParser("run.py")
parser.add_argument("device", help="The device to run experiments on", type=str, default="cuda", nargs="?")
args = parser.parse_args()

# Extract the absolute path of the data directory
data_dir = Path(__file__).resolve().parent / "data"

# Initialize a Ray instance
#init(num_gpus=1, num_cpus=16)

# ----------------------------------------------------------------------------------------------------------------

num_samples = 1

train_path = "adtof/adtof_yt_train"
val_path = "adtof/adtof_yt_validation"

config = {
    "batch_size": tune.choice([8, 16, 32]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "weight_decay": tune.loguniform(1e-2, 1e-4),
    "amsgrad": tune.choice([True, False]),
}

Model = ADTOF_FrameRNN

print(f"Main: Can use CUDA: {torch.cuda.is_available()}")

device = args.device
seed = int(time())

scheduler = ASHAScheduler(
    metric="loss",
    mode="min"
)

# Run the experiments
result = tune.run(
    partial(train, Model=Model, train_path=data_dir/train_path, val_path=data_dir/val_path, device=device, seed=seed),
    config=config,
    num_samples=num_samples,
    scheduler=scheduler,
    resources_per_trial={"gpu": 1}
)

# Print the results
best_trial = result.get_best_trial()
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final validation loss: {best_trial.last_result['loss']}")