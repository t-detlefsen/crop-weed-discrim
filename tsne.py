import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D
import argparse
import os

if __name__ == "__main__":
    # Handle experiment argument
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', metavar='DIRECTORY', help="Experiment to view TSNE on")
    args, opts = parser.parse_known_args()

    if args.experiment == None:
        raise Exception("No experiment specified")
    
    experiment_dir = os.path.join("exp", args.experiment)

    if not os.path.exists(experiment_dir):
        raise Exception("Experiment path not found")

    # Load features, labels, and class names
    features = np.load(experiment_dir + '/features.npy')  
    labels = np.load(experiment_dir + '/labels.npy')      
    class_names = np.load(experiment_dir + '/class_names.npy', allow_pickle=True)  

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    # Plot the t-SNE results
    plt.figure(figsize=(12, 8), constrained_layout = True)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab20', alpha=0.7)

    # Add legend with class names
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=scatter.cmap(scatter.norm(i)), label=class_name, markersize=10)
        for i, class_name in enumerate(class_names)
    ]

    plt.legend(handles=legend_elements, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')


    # Add plot details
    plt.title('t-SNE Visualization of Features')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()

    # Show the plot
    plt.show()
