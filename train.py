import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from ray import train, tune

from datasets import ADTOF_load

from pathlib import Path
from typing import Optional


def train(config: tune.TuneConfig, Model: nn.Module, train_path: Path, val_path: Path, device: str = "cpu", seed: Optional[int] = None):
    """ Training function to use with RayTune """

    # Declare device
    device = device if torch.cuda.is_available() else "cpu"

    # Load the datasets
    train_loader = ADTOF_load(train_path, batch_size=config["batch_size"], shuffle=True, seed=seed)
    val_loader = ADTOF_load(val_path, batch_size=config["batch_size"], shuffle=True, seed=seed)

    # Create the model, loss function and optimizer
    model = Model().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])
    optimizer.zero_grad(set_to_none=True)

    # Start training
    print(f"Started training on {device}")
    for epoch in range(3):
        model.train()
        train_loss = 0.0
        for i, data in enumerate(train_loader):
            # Perform forward, backward and optimization step
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)

            loss = loss_fn(outputs, labels).mean()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Print statistics every 2000th mini-batch
            train_loss += loss.item()
            if True or (i+1) % 2000 == 0:
                print(f"[Epoch {epoch+1}, {i+1}] loss: {train_loss / (i+1) :.4f}")

        # After a training epoch, compute validation performance
        model.eval()
        val_loss = 0.0
        for i, data in enumerate(val_loader):
            with torch.no_grad():
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)

                loss = loss_fn(outputs, labels).mean()
                val_loss += loss.item()
        
        # Report to RayTune
        train.report({"loss": val_loss / (i+1)})
    print("Finished training")
