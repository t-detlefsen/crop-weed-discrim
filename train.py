from crop_weed_discrim import trainer
import torch

from crop_weed_discrim.utils import ARGS
from crop_weed_discrim.models.resnet_cnn import ResNetCNN
from crop_weed_discrim.utils.dataloader import PlantSeedlingsDataset
import numpy as np
import torchvision
import torch.nn as nn
import random
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

# TODO: Add terminal output for important things (hyperparams)

if __name__ == "__main__":
    # TODO: Set random seeds

    np.random.seed(0)
    torch.manual_seed(0)
    random.seed(0)
    # TODO: Load in parameters from config.yaml
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    print(f"Loaded config parameters: {config}")

    args = ARGS(
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        test_batch_size=config.get('test_batch_size', config['batch_size']),
        lr=config['learning_rate'],
        step_size=config['step_size'],
        gamma=config['gamma'],
        log_every=config['log_every'],
        val_every=config['val_every'],
        save_freq=config['save_freq'],
        save_at_end=config.get('save_at_end', True),
        inp_size=config['input_size'],
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    # TODO: Initialize model
    model = ResNetCNN(len(PlantSeedlingsDataset.CLASS_NAMES)).to(args.device)
    torch.save(model.state_dict(), 'model.pth')

    # TODO: Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # TODO: Load in dataset + augmentations
    ### not sure if i am doing this correctly
    # Basic agumentation
    train_tfs = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
    val_tfs = []

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