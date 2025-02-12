import torch
from torch import nn, optim
import torch.nn.functional as F

from ray import train, tune
from ray.train import Checkpoint

from preprocess import compute_infrequency_weights
from evaluate import compute_peaks, compute_predictions, f_measure

from pathlib import Path
from copy import deepcopy


def train_model(config: tune.TuneConfig):
    """ Training function to use with RayTune """

    # Declare device
    device = config["device"] if torch.cuda.is_available() else "cpu"
    print(f"Training: Can use CUDA: {torch.cuda.is_available()}")
        
    # Load the datasets
    train_loader = config["train_loader"]
    val_loader = config["val_loader"]

    # Create the model, loss function and optimizer
    model = config["Model"](**config["parameters"]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = config["optimizer"](model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"], amsgrad=config["amsgrad"])
    optimizer.zero_grad(set_to_none=True)

    # Add a learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5)

    # Compute infrequent instrument weights from the training dataset
    infrequency_weights = compute_infrequency_weights(train_loader).to(device)
    print("Infrequency weights: ", infrequency_weights)
    
    # Start training
    print(f"Started training on {device}")
    epochs_since_improvement, val_loss_best, val_f1_micro_best, model_val_state_dict = 0, None, None, None
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss, n_batches_train = 0.0, 0
        for i, data in enumerate(train_loader):
            # Perform forward, backward and optimization step
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)

            # Compute class weights given labels and infrequency weights
            class_weights = torch.where(labels == 0, torch.tensor(0.0), infrequency_weights).sum(dim=2)
            class_weights = torch.where(class_weights == 0.0, torch.tensor(1.0), class_weights)

            loss = (loss_fn(outputs, labels).sum(dim=2) * class_weights).mean()
            loss.backward()

            # Clip the gradients to prevent explosions
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            # And store training loss
            train_loss += loss.item()
            n_batches_train += 1


        # After a training epoch, compute validation performance
        model.eval()
        val_loss, n_batches_val = 0.0, 0
        val_predictions = torch.zeros(size=(infrequency_weights.shape[0], 3))
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)

                class_weights = torch.where(labels == 0, torch.tensor(0.0), infrequency_weights).sum(dim=2)
                class_weights = torch.where(class_weights == 0.0, torch.tensor(1.0), class_weights)

                loss = (loss_fn(outputs, labels).sum(dim=2) * class_weights).mean()
                val_loss += loss.item()
                n_batches_val += 1

                # Compute activation probabilties and add to predictions
                activations = F.sigmoid(outputs)
                val_predictions += compute_predictions(compute_peaks(activations), labels)

                if i == 0:
                    print(torch.stack((F.sigmoid(outputs), labels), dim=-1))
                    
        # Average the losses
        train_loss /= n_batches_train
        val_loss /= n_batches_val

        # Step the scheduler
        scheduler.step(val_loss)
        print("Learning rate:", scheduler.get_last_lr())

        # Compute F1 score
        val_f1_micro, val_f1_macro, val_f1_class = f_measure(val_predictions)
        print("Predictions:", val_predictions)
        print("Class F1s:", val_f1_class)

        # Check if we should increment early stop count
        if val_loss_best is None or val_loss < val_loss_best:
            val_loss_best = val_loss
            epochs_since_improvement = 0

            model_val_state_dict = deepcopy(model.state_dict())
        else:
            epochs_since_improvement += 1
            
        # Check if we should checkpoint current model by comparing micro F1 score
        checkpoint = None
        if val_f1_micro_best is None or val_f1_micro > val_f1_micro_best:
            val_f1_micro_best = val_f1_micro

            # Create a checkpoint directory within the trial directory
            trial_dir = train.get_context().get_trial_dir()
            checkpoint_dir = Path(trial_dir) / f"checkpoint_epoch_{epoch}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model to the checkpoint directory
            model_path = checkpoint_dir / "model.pt"
            torch.save(model.state_dict(), model_path)

            # Also save model with lowest validation loss
            model_val_path = checkpoint_dir / "model_val.pt"
            torch.save(model_val_state_dict, model_val_path)
            
            # Create a Checkpoint object
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
        
        # Report to RayTune
        train.report({
            "Training Loss": train_loss,
            "Validation Loss": val_loss,
            "Micro F1": val_f1_micro.item(),
            "Macro F1": val_f1_macro.item(),
            "Class F1": val_f1_class.tolist(),
            "epochs_since_improvement": epochs_since_improvement
            },
            checkpoint=checkpoint
        )
    print("Finished training")
