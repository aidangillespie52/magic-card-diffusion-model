from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import os

class MTGCardsDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Directory path containing subfolders of different set names.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        # Assuming the folder structure inside `data_dir` is as follows:
        # data/
        #   set_name_1/
        #     card_1.jpg
        #     card_2.jpg
        #     ...
        #   set_name_2/
        #     card_1.jpg
        #     card_2.jpg
        #     ...
        # The folder names (set_name_1, set_name_2, ...) will be treated as classes
        
        # Create the ImageFolder dataset object
        self.data = ImageFolder(root=data_dir, transform=transform, allow_empty=True)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes