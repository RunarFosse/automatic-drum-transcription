import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ray import train, tune

from datasets import ADTOF_load
from loss import compute_infrequency_weights

from pathlib import Path
from typing import Optional

from time import sleep, time


def train_model(config: tune.TuneConfig, Model: nn.Module, n_epochs: int, train_path: Path, val_path: Path, device: str = "cpu", seed: Optional[int] = None):
    """ Training function to use with RayTune """

    # Declare device
    device = device if torch.cuda.is_available() else "cpu"
    print(f"Training: Can use CUDA: {torch.cuda.is_available()}")
        
    # Load the datasets
    train_loader = ADTOF_load(train_path, batch_size=config["batch_size"], shuffle=False, seed=seed)
    val_loader = ADTOF_load(val_path, batch_size=config["batch_size"], shuffle=False, seed=seed)

    # Create the model, loss function and optimizer
    model = Model().to(device)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])
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

            print(outputs.shape)
            print(labels.shape)
            loss = loss_fn(outputs, labels)
            print(loss)
            print(loss.mean())
            print(loss.sum())
            #print(torch.stack((loss, class_weights), dim=-1))
            loss.mean().backward()

            optimizer.step()
            optimizer.zero_grad()

            # And store training loss
            train_loss += loss.item()
            n_batches_train += 1
        
        print(torch.stack((outputs, labels), dim=-1))

        # After a training epoch, compute validation performance
        model.eval()
        val_loss, val_f1_micro, val_f1_macro = 0.0, 0.0, 0.0
        n_batches_val = 0
        for i, data in enumerate(val_loader):
            with torch.no_grad():
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)

                loss = loss_fn(outputs, labels).mean()
                val_loss += loss.item()
                n_batches_val += 1

                # Compute F1 score over frames and batches (TODO!)
                #frames = labels.shape[1]
                #for frame in range(frames):
                #    val_f1_micro += multiclass_f1_score(outputs[:, frame], labels[:, frame], num_classes=5, average="micro").mean() / frames
                #    val_f1_macro += multiclass_f1_score(outputs[:, frame], labels[:, frame], num_classes=5, average="macro").mean() / frames
        
        # Report to RayTune
        train.report({
            "Training Loss": train_loss / n_batches_train,
            "Validation Loss": val_loss / n_batches_val,
            })
    print("Finished training")
