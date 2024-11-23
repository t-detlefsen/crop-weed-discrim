import os
import shutil

import torch
from torch.utils.tensorboard import SummaryWriter

class ARGS(object):
    """
    Tracks hyper-parameters for trainer code 
        - Feel free to add your own hparams below (cannot have __ in name)
        - Constructor will automatically support overrding for non-default values
    """
    # Model
    model_name = "resnet_cnn"

    # Dataset
    dataset_name = "plant_seedlings"

    # Hyperparameters
    epochs = 50 # Number of epochs to train for
    batch_size = 256 # Batch size for training
    lr = 5e-4 # Learning rate
    step_size = 6 # Learning rate step
    gamma = 0.25 # Learning rate gamma
    use_cuda = True # Enable GPU during trainingß
    val_every = 5 # Validate very N epochs

    # Pairwise Confusion
    pc_loss = True
    lambda_pc = 0.1  # Weight for Pairwise Confusion
    lambda_ec = 0.01 # Weight for Entropic Confusion

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            assert '__' not in k and hasattr(self, k), "invalid attribute!"
            assert k != 'device', "device property cannot be modified"
            setattr(self, k, v)

    def __repr__(self):
        repr_str = ''
        for attr in dir(self):
            if '__' not in attr and attr !='use_cuda':
                repr_str += 'args.{} = {}\n'.format(attr, getattr(self, attr))
        return repr_str
    
    @property
    def device(self):
        return torch.device("cuda" if self.use_cuda else "cpu")
    
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

####################################################################################################
# Pairwise Confusion and Entropic Confusion Losses
# Refer to https://github.com/abhimanyudubey/confusion

def PairwiseConfusion(features):
    batch_size = features.size(0)
    if batch_size % 2 != 0:
        features = features[:-1]  # Drop the last sample to make the batch size even
    batch_left = features[:batch_size // 2]
    batch_right = features[batch_size // 2:]
    loss = torch.norm((batch_left - batch_right).abs(), 2, 1).sum() / float(batch_size)

    return loss

def EntropicConfusion(features):
    batch_size = features.size(0)

    return torch.mul(features, torch.log(features)).sum() * (1.0 / batch_size)

def PairwiseEntropicConfusionLoss(args, features, output, labels, loss_fn):
    classification_loss = loss_fn(output, labels)
    pairwise_confusion_loss = PairwiseConfusion(features)
    entropic_confusion_loss = EntropicConfusion(torch.softmax(output, dim=1))
    
    total_loss = (
        classification_loss
        + args.lambda_pc * pairwise_confusion_loss
        + args.lambda_ec * entropic_confusion_loss
    )

    return total_loss
####################################################################################################