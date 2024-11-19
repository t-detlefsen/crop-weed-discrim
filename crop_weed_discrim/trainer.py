from __future__ import print_function

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from crop_weed_discrim.utils.dataloader import PlantSeedlingsDataset  

import os
from datetime import datetime

def create_experiment_directory(args):
    """
    Makes a directory based on hyperparameters and timestamp.
    """
    # Include hyperparameters in the experiment name
    experiment_name = (
        f"lr{args.lr}_bs{args.batch_size}_ss{args.step_size}_gamma{args.gamma}_inp{args.inp_size}_"
        + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )

    # Create the directory
    experiment_dir = os.path.join("exp", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"Experiment directory created: {experiment_dir}")
    return experiment_dir



def save_model(epoch, model_name, model):
    """
    From Assigment
    """
    filename = f'checkpoint-{model_name}-epoch{epoch+1}.pth'
    print(f"Saving model at {filename}")
    torch.save(model.state_dict(), filename)


def save_this_epoch(args, epoch):
    """
    From assignment
    """
    if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch + 1) == args.epochs:
        return True
    return False


def train(args, model, optimizer, scheduler=None, model_name="model"):
    """
    Train the model based on specified training parameters.

    Args:
        args: Training arguments including hyperparameters and configurations.
        model: The model to be trained.
        optimizer: Optimizer for training the model.
        scheduler: Learning rate scheduler (optional).
        model_name: Name to save the model under.

    Returns:
        None
    """
    # TODO: Create a new experiment under the exp directory (unique name, save hyperparameters, tensorboard)
    experiment_dir = create_experiment_directory()
    writer = SummaryWriter(log_dir=experiment_dir)
    print(f"Experiment logs saved at {experiment_dir}")

    # Get data loaders (assuming utils provides `get_data_loader`)
    train_loader = utils.get_data_loader(
        dataset='plant_seedlings', train=True, batch_size=args.batch_size, split='train', inp_size=args.inp_size)
    val_loader = utils.get_data_loader(
        dataset='plant_seedlings', train=False, batch_size=args.test_batch_size, split='val', inp_size=args.inp_size)

    # Ensure model is on the correct device
    model.train()
    model = model.to(args.device)
    cnt = 0  # Counter for logging

    for epoch in range(args.epochs):

        # TODO: Train model
        epoch_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):

            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)

            # Forward pass
            optimizer.zero_grad()
            output = model(data)

            # Cross entropy loss with pytorch
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            # Add loss to tensorboard
            writer.add_scalar("Loss/train", loss.item(), cnt)

            # TODO: Add terminal output for important things (acc, loss, etc.)
            if cnt % args.log_every == 0:
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}")

            cnt += 1

        print(f"Epoch {epoch+1} | Average Training Loss: {epoch_loss / len(train_loader):.4f}")

        # TODO: Validate model (every N epochs)
        if epoch % args.val_every == 0:
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(args.device), target.to(args.device)
                    output = model(data)
                    val_loss += loss_fn(output, target).item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()

            val_loss /= len(val_loader)
            accuracy = correct / len(val_loader.dataset)
            print(f"Validation Epoch {epoch+1} | Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
            writer.add_scalar("Validation/Loss", val_loss, cnt)
            writer.add_scalar("Validation/Accuracy", accuracy, cnt)
            model.train()

        # TODO: Save results
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], cnt)

    writer.close()