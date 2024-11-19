import os
import shutil
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def create_experiment_directory():
    '''
    Create a new experiment directory w/ configs and tensorboard writer

    Returns:
        experiment_dir (str): The experiment directory to save files
        label (torch.utils.tensorboard.writer.SummaryWriter): Tensorboard writer
    '''
    # Find newest experiment
    exps = os.listdir("exp")
    exps = [exp for exp in exps if (exp != '.DS_Store' and exp != '.gitignore')]
    try:
        experiment_name = "exp" + str(max([int(exp[3:]) for exp in exps]) + 1)
    except:
        experiment_name = "exp0"

    # Create the directory
    experiment_dir = os.path.join("exp", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"Experiment directory created: {experiment_dir}")

    # Create files
    shutil.copyfile("config.yaml", os.path.join(experiment_dir, "config.yaml")) # Configs
    writer = SummaryWriter(log_dir=experiment_dir) # Tensorboard writer

    return experiment_dir, writer

def train(args, data, model, loss_fn, optimizer, scheduler=None):
    """
    Train the model based on specified training parameters.

    Args:
        args (ARGS): Training arguments including hyperparameters and configurations.
        data (dict<SubsetWrapper>): Pre-split train, test, and validation data
        model (nn.Module): The model to train
        loss_fn (torch.nn.modules.loss): Loss function
        optimizer (torch.optim.adam.Adam): Optimizer
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler

    Returns:
        experiment_dir (str): The experiment directory to save files
    """
    # Create a new experiment under the exp directory (unique name, save hyperparameters, tensorboard)
    experiment_dir, writer = create_experiment_directory()

    # Create dataloaders
    train_loader = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(data["val"], batch_size=args.batch_size, shuffle=True)

    # Ensure model is on the correct device
    model.train()
    model = model.to(args.device)

    for epoch in range(args.epochs):
        epoch_loss = 0
        best_acc = 0

        # Train model
        print(f"Train Epoch {epoch}")
        for images, labels in tqdm(train_loader):
            images, labels = images.to(args.device), labels.to(args.device)

            # Forward pass
            optimizer.zero_grad()
            output = model(images)

            # Calculate loss
            loss = loss_fn(output, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Report stats
        print(f"Epoch {epoch} | Average Training Loss: {epoch_loss / len(train_loader):.4f}")
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], epoch)

        # Validate model
        if epoch % args.val_every == 0:
            print(f"Val Epoch {epoch}")
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for image, label in tqdm(val_loader):
                    image, label = image.to(args.device), label.to(args.device)
                    output = model(image)
                    val_loss += loss_fn(output, label).item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(label).sum().item()

            val_loss /= len(val_loader)
            accuracy = correct / len(val_loader.dataset)
            print(f"Validation Epoch {epoch+1} | Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
            writer.add_scalar("Validation/Loss", val_loss, epoch)
            writer.add_scalar("Validation/Accuracy", accuracy, epoch)

            # Save results
            if accuracy > best_acc:
                print("Saving model")
                torch.save(model.state_dict(), os.path.join(experiment_dir, "weights.pth"))
                best_acc = accuracy

        # Step the scheduler
        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)

    writer.close()

    return experiment_dir