import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

class PlantSeedlingsDataset(Dataset):
    '''
    Dataloader for the Plant Seedlings dataset Nonsegmented Single Plants from Aarhus University

    https://vision.eng.au.dk/plant-seedlings-dataset/#citation
    '''
    def __init__(self, data_dir='data/plant_seedlings_dataset/'):
        '''
        Initialize the PlantSeedlingsDataset object

        Args:
            data_dir (str): The location of the dataset
        '''
        super().__init__()
        self.data_dir = data_dir
        
        # Determine classes
        self.class_names = os.listdir(self.data_dir)
        self.class_names = [folder for folder in self.class_names if folder != '.DS_Store']
        self.class_names.sort()

        # Create image list
        self.image_list = []
        for class_name in self.class_names:
            file_list = os.listdir(self.data_dir+class_name)
            file_list = [class_name + "/" + file for file in file_list if file != '.DS_Store']
            
            self.image_list += file_list

    def __getitem__(self, idx):
        '''
        Retrieve an image and label from the dataset

        Args:
            idx (int): The index of the item to be retrieved

        Returns:
            image (torch.tensor): The image corresponding to the index
            label (int): The label corresponding to the index
        '''
        label = self.class_names.index(self.image_list[idx].split("/")[0])

        image = Image.open(self.data_dir + self.image_list[idx])

        # Remove image transparency if necessary
        if image.mode == 'RGBA':
            image.load()
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        
        return image, label

    def __len__(self):
        '''
        Return the length of the dataset

        Returns:
            length (int): the length of the dataset
        '''
        length = len(self.image_list)
        return length
    
class SubsetWrapper(Dataset):
    '''
    A wrapper to apply transforms to a subset of data
    '''

    def __init__(self, subset, size=224, tfs=[]):
        '''
        Initialize the PlantSeedlingsDataset object

        Args:
            subset (torch.utils.dataset.Subset): The subset of data to wrap
            size (int): The size of the input images
            tfs (list[torchvision.transforms.transform]): A list of transforms to apply to the image
        '''
        self.subset = subset
        self.size = size

        # Create transforms
        self.transforms = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            *tfs,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.457, 0.407], std=[0.5, 0.5, 0.5])])
        
    def __getitem__(self, idx):
        '''
        Retrieve an image and label from the dataset

        Args:
            idx (int): The index of the item to be retrieved

        Returns:
            image (torch.tensor): The image corresponding to the index
            label (int): The label corresponding to the index
        '''
        image, label = self.subset[idx]
        
        image = self.transforms(image)

        return image, label
    
    def __len__(self):
        '''
        Return the length of the dataset

        Returns:
            length (int): the length of the dataset
        '''
        return len(self.subset)