import os
from tqdm import tqdm
from sklearn import metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
from torch.utils.data import DataLoader

def test(args, data, model, experiment_dir, class_names):
    """
    Determine the accuracy, generate confusion matrix, and tsne plot given a testset and model

    Args:
        args (ARGS): Training arguments including hyperparameters and configurations.
        data (dict<SubsetWrapper>): Pre-split train, test, and validation data
        model (nn.Module): The model to train
        experiment_dir (str): The experiment directory to save files
    """
    
    # Create dataloader
    test_loader = DataLoader(data["test"], batch_size=1, shuffle=False, num_workers=0)

    # Initialize arrays for storing features and labels
    all_features = []
    all_labels = []

    # Freeze weights
    with torch.no_grad():
        model.eval()
        correct = 0
        predicted = []
        actual = []

        # Loop through each image in test set
        for images, labels in tqdm(test_loader):
            images, labels = images.to(args.device), labels.to(args.device)

            features, logits = model(images)  # Unpack the tuple
            logits = logits.unsqueeze(0)  # Add a batch dimension
            pred = logits.argmax(dim=1)  # Use logits to calculate predictions

            all_features.append(features.cpu().detach())
            all_labels.append(labels.cpu().detach())

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

        # Plot TSNE Visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(torch.vstack(all_features))

        plt.figure(figsize=(12, 8), constrained_layout = True)
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=torch.vstack(all_labels), cmap='tab20', alpha=0.7)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(i)), label=class_name, markersize=10)
            for i, class_name in enumerate(class_names)
        ]

        plt.legend(handles=legend_elements, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.title('t-SNE Visualization of Features')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.tight_layout()

        plt.savefig(os.path.join(experiment_dir, "tsne.png"))