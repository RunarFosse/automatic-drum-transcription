import argparse
import torch
from torch import optim
from ray import init, tune
from time import time
from models import ADTOF_FrameRNN, ADTOF_FrameAttention
from train import train_model
from pathlib import Path
import tensorflow as tf

# Only run this file directly
assert __name__ == "__main__"

# Declare an argument parser for this file
parser = argparse.ArgumentParser("run.py")
parser.add_argument("device", help="The device to run experiments on", type=str, default="cuda", nargs="?")
args = parser.parse_args()

# Extract the absolute path of the data directory
data_dir = Path(__file__).resolve().parent / "data"

# Disable all GPUs for TensorFlow
tf.config.set_visible_devices([], 'GPU')

# Initialize a Ray instance
init(num_gpus=1, num_cpus=16)

# ----------------------------------------------------------------------------------------------------------------

study = "Architectural Performance"
experiment = "Convolutional Transformer"
dataset = "ADTOF-YT"

Model = ADTOF_FrameAttention

num_samples = 10
num_epochs = 100

train_path = "adtof/adtof_yt_train"
val_path = "adtof/adtof_yt_validation"

print(f"Main: Can use CUDA: {torch.cuda.is_available()}")

device = args.device
seed = int(time())

config = {
    "batch_size": tune.choice([128, 256]),

    "lr": tune.loguniform(1e-5, 1e-3),
    "weight_decay": tune.loguniform(1e-5, 1e-2),
    "amsgrad": tune.choice([True, False]),
    "optimizer": optim.AdamW,

    "Model": Model,
    "n_epochs": num_epochs,

    "device": device,
    "seed": seed,
}

# Run the experiments
result = tune.run(
    tune.with_parameters(train_model, train_path=data_dir/train_path, val_path=data_dir/val_path),
    config=config,
    num_samples=num_samples,
    resources_per_trial={"gpu": 1, "accelerator_type:A100": 1},
    stop={"epochs_since_improvement": 10},
    keep_checkpoints_num=1,
)

# Print the results
best_trial = result.get_best_trial("Global F1", mode="min", scope="all")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final validation loss: {best_trial.last_result['Validation Loss']}")
print(f"Best trial final validation global F1: {best_trial.last_result['Global F1']}")
print(f"Best trial final validation class F1: {best_trial.last_result['Class F1']}")

"""train_model({
    "batch_size": 1,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "amsgrad": True,
}, Model=Model, n_epochs=100, train_path=data_dir/train_path, val_path=data_dir/val_path, device=device, seed=seed)"""