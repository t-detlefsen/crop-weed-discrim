import os
import yaml
import random
import argparse
import numpy as np

import torch
from torchvision import transforms

from crop_weed_discrim import trainer, tester
from crop_weed_discrim.utils.utils import ARGS
# Import ResNetCNN that outputs both features and logits for PC
from crop_weed_discrim.models.resnet_pc import ResNetCNN
from crop_weed_discrim.utils.dataloader import PlantSeedlingsDataset, SubsetWrapper

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', metavar='DIRECTORY', help="Experiment to run test on")
    args, opts = parser.parse_known_args()

    if args.experiment == None:
        raise Exception("No experiment specified")
    
    experiment_dir = os.path.join("exp", args.experiment)

    if not os.path.exists(experiment_dir):
        raise Exception("Experiment path not found")

    # Set random seeds
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # Load in parameters from config.yaml
    with open(os.path.join(experiment_dir, "config.yaml"), "r") as config_file:
        config = yaml.safe_load(config_file)

    configs = ARGS(
        model_name = config['model']['name'],
        dataset_name = config['dataset']['name'],
        epochs = config['hyperparams']['epochs'],
        batch_size = config['hyperparams']['batch_size'],
        lr = config['hyperparams']['lr'],
        step_size = config['hyperparams']['step_size'],
        gamma = config['hyperparams']['gamma'],
        use_cuda = config['hyperparams']['use_cuda'],
        val_every = config['hyperparams']['val_every']
    )

    print("Loaded config.yaml:")
    print(configs)

    # Load dataset
    datasets = ["plant_seedlings"]
    if configs.dataset_name == "plant_seedlings":
        dataset = PlantSeedlingsDataset()
    else:
        raise NotImplementedError(f"{configs.dataset_name} not implemented, choose one of the following {datasets}")
    
    # Create Augmentations
    tfs = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=45),
        transforms.ColorJitter(),
    ]

    # Split into train, val, test
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [round(0.8 * len(dataset)), round(0.1 * len(dataset)), round(0.1 * len(dataset))])
    data = {"train": SubsetWrapper(train_set, tfs=tfs),
            "val": SubsetWrapper(val_set),
            "test": SubsetWrapper(test_set)}

    # Initialize model
    model = ResNetCNN(len(dataset.class_names))

    # Load model
    model.load_state_dict(torch.load(os.path.join(experiment_dir, 'weights.pth'), map_location=configs.device))
    model = model.to(configs.device)

    # Test model using tester.py
    tester.test(configs, data, model, experiment_dir, dataset.class_names)