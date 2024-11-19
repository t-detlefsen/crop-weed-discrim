import torch

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