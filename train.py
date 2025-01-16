import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from ray import train, tune

from datasets import ADTOF_load
from preprocess import compute_infrequency_weights
from evaluate import compute_peaks, compute_predictions, f_measure

from pathlib import Path
from typing import Optional

from time import sleep, time


def train_model(config: tune.TuneConfig, Model: nn.Module, n_epochs: int, train_path: Path, val_path: Path, device: str = "cpu", seed: Optional[int] = None):
    """ Training function to use with RayTune """

    # Declare device
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Training: Can use CUDA: {torch.cuda.is_available()}")
        
    # Load the datasets
    train_loader = ADTOF_load(train_path, batch_size=config["batch_size"], shuffle=True, seed=seed)
    val_loader = ADTOF_load(val_path, batch_size=config["batch_size"], shuffle=False, seed=seed)

    # Create the model, loss function and optimizer
    model = Model().to(device)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = config["optimizer"](model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])
    optimizer.zero_grad(set_to_none=True)

    # Compute infrequent instrument weights from the training dataset
    infrequency_weights = compute_infrequency_weights(train_loader).to(device)
    print("Infrequency weights: ", infrequency_weights)
    
    # Start training
    print(f"Started training on {device}")
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        n_batches_train = 0
        for i, data in enumerate(train_loader):
            # Perform forward, backward and optimization step
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)

            # Compute class weights given labels and infrequency weights
            class_weights = torch.where(labels == 0, torch.tensor(1.0), infrequency_weights)

            #print(torch.stack((loss_fn(outputs, labels), class_weights), dim=-1))

            loss = (loss_fn(outputs, labels) * class_weights).sum(dim=(1, 2)).mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # And store training loss
            train_loss += loss.item()
            n_batches_train += 1
        
        print(torch.stack((F.sigmoid(outputs), labels), dim=-1))

        # After a training epoch, compute validation performance
        model.eval()
        val_loss = 0.0
        val_predictions = None
        n_batches_val = 0
        for i, data in enumerate(val_loader):
            with torch.no_grad():
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)

                loss = loss_fn(outputs, labels).sum(dim=(1, 2)).mean()
                val_loss += loss.item()
                n_batches_val += 1

                # Add to predictions
                if val_predictions is None:
                    val_predictions = compute_predictions(compute_peaks(outputs), labels)
                else:
                    val_predictions += compute_predictions(compute_peaks(outputs), labels)
                    
        # Compute F1 score
        print(val_predictions)
        val_f1_global, val_f1_class = f_measure(val_predictions)
        
        # Report to RayTune
        train.report({
            "Training Loss": train_loss / n_batches_train,
            "Validation Loss": val_loss / n_batches_val,
            "Global F1": val_f1_global.item() / n_batches_val,
            "Class F1": [round(f1 / n_batches_val, 3) for f1 in val_f1_class],
            })
    print("Finished training")
