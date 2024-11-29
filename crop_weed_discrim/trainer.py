import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from crop_weed_discrim.utils.utils import create_experiment_directory, PairwiseEntropicConfusionLoss

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
    best_acc = 0

    # Create dataloaders
    train_loader = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0) # Drop last to ensure even batch size
    val_loader = DataLoader(data["val"], batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Ensure model is on the correct device
    model.train()
    model = model.to(args.device)

    # Initialize arrays for storing features and labels
    all_features = []
    all_labels = []

    for epoch in range(args.epochs):
        epoch_loss = 0

        # Train model
        print(f"Train Epoch {epoch}")
        for images, labels in tqdm(train_loader):
            images, labels = images.to(args.device), labels.to(args.device)

            # Forward pass
            optimizer.zero_grad()
            features, output = model(images)

            #  Store features and labels
            all_features.append(features.cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())

            # Calculate loss
            if args.pc_loss:
                loss = PairwiseEntropicConfusionLoss(args, features, output, labels, loss_fn)
            else:
                loss = loss_fn(output, labels)

            # Backward pass
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        # Report stats
        print(f"Epoch {epoch} | Average Training Loss: {epoch_loss / len(train_loader):.4f}")
        writer.add_scalar("Train/Loss", epoch_loss, epoch)

        # Validate model
        if epoch % args.val_every == 0:
            print(f"Val Epoch {epoch}")
            model.eval()
            val_loss = 0
            correct = 0
            # Validation loop
            with torch.no_grad():
                for image, label in tqdm(val_loader):
                    image, label = image.to(args.device), label.to(args.device)
                    features, output = model(image)
                    val_loss += loss_fn(output, label).item() # Use logits for loss calculation
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

    # Save features and labels
    np.save(experiment_dir +'/features.npy', np.vstack(all_features))  
    np.save(experiment_dir +'/labels.npy', np.hstack(all_labels))  

    return experiment_dir