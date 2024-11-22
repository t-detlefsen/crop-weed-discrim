import os
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

# TODO: Evaluate accuracy on data["test"]
# TODO: Viusualize confusion matrix for data["test"]

def test(args, data, model, experiment_dir, class_names):
    """
    Determine the accuracy and confusion matrix given a testset and model

    Args:
        args (ARGS): Training arguments including hyperparameters and configurations.
        data (dict<SubsetWrapper>): Pre-split train, test, and validation data
        model (nn.Module): The model to train
        experiment_dir (str): The experiment directory to save files
    """
    # Create dataloader
    test_loader = DataLoader(data["test"], batch_size=1, shuffle=False)

    # Freeze weights
    with torch.no_grad():
        model.eval()
        correct = 0
        predicted = []
        actual = []

        # Loop through each image in test set
        for images, labels in tqdm(test_loader):
            images, labels = images.to(args.device), labels.to(args.device)
            # output = model(images)
            # pred = output.argmax()

            ####################################################################################################
            __, logits = model(images)  # Unpack the tuple
            logits = logits.unsqueeze(0)  # Add a batch dimension
            pred = logits.argmax(dim=1)  # Use logits to calculate predictions  
            ####################################################################################################
            correct += pred.eq(labels).sum().item()
            predicted.append(pred.item())
            actual.append(labels.item())

        # Record Final Accuracy
        print(f"Final Accuracy {correct / len(test_loader.dataset)}")

        # Save Accuracy and Confusion Matrix to Terminal
        fig = plt.figure(figsize=[10, 10])

        confusion_matrix = metrics.confusion_matrix(actual, predicted)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = class_names)
        cm_display.plot()

        plt.title("Test Set Accuracy %.2f%% of %d images" % ((correct / len(test_loader.dataset)) * 100, len(test_loader.dataset)))        
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, "confusion_matrix.png"))