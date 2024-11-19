# from crop_weed_discrim import trainer
from crop_weed_discrim.utils.utils import ARGS
# from crop_weed_discrim.models.resnet_cnn import ResNetCNN
from crop_weed_discrim.utils.dataloader import PlantSeedlingsDataset, SubsetWrapper

import yaml
import random
import numpy as np

import torch
# import torchvision
# import torch.nn as nn
from torchvision import transforms
# from torch.utils.data import DataLoader

if __name__ == "__main__":
    # Set random seeds
    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)

    # Load in parameters from config.yaml
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    configs = ARGS(
        model_name = config['model']['name'],
        dataset_name = config['dataset']['name'],
        epochs = config['hyperparams']['epochs'],
        batch_size = config['hyperparams']['batch_size'],
        lr = config['hyperparams']['lr'],
        step_size = config['hyperparams']['step_size'],
        gamma = config['hyperparams']['gamma'],
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
    train_set = SubsetWrapper(train_set, tfs=tfs)
    val_set = SubsetWrapper(val_set)
    test_set = SubsetWrapper(test_set)

    import ipdb
    ipdb.set_trace()
    exit()

    # Initialize model
    model = ResNetCNN(len(PlantSeedlingsDataset.CLASS_NAMES)).to(args.device)
    torch.save(model.state_dict(), 'model.pth')

    # TODO: Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # Initialize datasets, no datset yet
    train_dataset = PlantSeedlingsDataset(size=args.inp_size, tfs=train_tfs, data_dir='data/')
    val_dataset = PlantSeedlingsDataset(size=args.inp_size, tfs=val_tfs, data_dir='data/')

    # Make data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=4)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # TODO: Setup loss function
    loss_function = torch.nn.CrossEntropyLoss() # do we need soemthing different???

    # TODO: Train model using trainer.py
    trainer.train(args, model, optimizer, scheduler)