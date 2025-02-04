import argparse
import torch
from torch import optim
from ray import init, tune, train
from time import time
from models import ADTOF_FrameRNN, ADTOF_FrameAttention
from evaluate import evaluate_model
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
root_dir = Path(__file__).resolve().parent
data_dir = root_dir / "data"

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
test_path = "adtof/adtof_yt_test"

print(f"Main: Can use CUDA: {torch.cuda.is_available()}")

device = args.device
seed = int(time())

config = {
    "batch_size": 128,

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
tuner = tune.Tuner(
    tune.with_resources(
        trainable=tune.with_parameters(train_model, train_path=data_dir/train_path, val_path=data_dir/val_path),
        resources={"gpu": 1, "accelerator_type:A100": 1}
    ),
    param_space=config,
    tune_config=tune.TuneConfig(
        num_samples=num_samples
    ),
    run_config=train.RunConfig(
        stop={"epochs_since_improvement": 5},
        checkpoint_config=train.CheckpointConfig(num_to_keep=1),
        verbose=2
    )
)
results = tuner.fit()

# Print the results
best_result = results.get_best_result("Global F1", mode="max", scope="all")
print(f"Best result config: {best_result.config}")
print(f"Best result final validation loss: {best_result.metrics['Validation Loss']}")
print(f"Best result final validation global F1: {best_result.metrics['Global F1']}")
print(f"Best result final validation class F1: {best_result.metrics['Class F1']}")

# Load the state_dict of the best performing model
checkpoint_path = Path(best_result.get_best_checkpoint("Global F1", mode="max").path)
state_dict = torch.load(checkpoint_path / "model.pt")

print("State_dict stored at:", best_result.get_best_checkpoint("Global F1", mode="max").path)
print(state_dict)
print(Model().load_state_dict(state_dict))

# Store the best performing model and its metrics to study/experiment path
model_path = (root_dir / "study" / study / experiment / dataset)
model_path.mkdir(parents=True, exist_ok=True)
torch.save(state_dict, model_path / "model.pt")
best_result.metrics_dataframe.to_csv(model_path / "metrics.csv")

# Load the best performing model and evaluate it on the test dataset
model = Model().load_state_dict(state_dict)
test_f1_global, test_f1_class = evaluate_model(model, test_path=test_path, device=device)

print(" ---------- Evaluation of best perfoming model ---------- ")
print(f"Global F1: {test_f1_global.item():.4f}")
print(f"Class F1: {[f'{test_f1.item():.4f}' for test_f1 in test_f1_class]}")