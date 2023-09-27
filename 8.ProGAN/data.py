"""Implements the DataLoading for Pro-GAN.
"""
import torch

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from glob import glob

from PIL import Image


class CelebADataset(Dataset):
    """Implements the CelebA Dataset for ProGAN."""

    def __init__(self, root_dir: str):
        """Initializes the CelebA Dataset.

        Parameters
        ----------
        root_dir : str
            The root directory of the dataset.
        """

        ## Setting the transforms ##
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: (x - 0.5) / 0.5),
            ]
        )

        ## Getting the image paths ##
        self.image_paths = glob(root_dir + "/*.jpg")

    def __getitem__(self, idx):
        """Returns a single datapoint from the dataset.

        Parameters
        ----------
        idx : int
            The index of the datapoint to be returned.
        """

        ## Opening the image ##
        image = Image.open(self.image_paths[idx])

        ## Transforming the image ##
        image = self.transform(image)

        return image, torch.LongTensor([0])

    def __len__(self):
        """Returns the length of the dataset."""

        return len(self.image_paths)
