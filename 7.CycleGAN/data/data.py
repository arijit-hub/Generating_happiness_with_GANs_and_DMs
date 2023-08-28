"""Implements the dataset."""

import lightning.pytorch as pl
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

from PIL import Image
import os
from glob import glob
import pandas as pd


def get_images(
    csv_path: str,
    root_img_path: str,
    split: str = "train",
    img_type: str = "normal",
    train_split_values: list = [0, 1, 2],
):
    """Given the csv_path returns the list of images in PIL
    in the split.

    Parameters
    ----------
    csv_path : str
        Path to the csv file.

    root_img_path: str
        Path to the root folder containing the images.

    split : str
        Split to get the images from. Can be train, test or val.
        [Options : "train", "test", "val"]
        [Default : "train"]

    img_type : str
        Type of images to get. Can be normal, cone or all.
        [Options : "normal", "cone", "all"]
        [Default : "normal"]

    train_split_values : list
        List of values to be considered as train split.
        [Default : [0,1,2]]
    """

    ## Read the csv file ##
    df = pd.read_csv(csv_path)

    ## Filter the dataframe based on the split ##

    if split == "train":
        df = df[df["split"].isin(train_split_values)]

    elif split == "val":
        df = df[~((df["split"].isin(train_split_values)) | (df["split"] == 4))]

    elif split == "test":
        df = df[df["split"] == 4]

    ## Filter the dataframe based on the type ##

    if img_type == "normal":
        normal_organs = ["breast", "carotid", "thyroid"]
        df = df[df["organ"].isin(normal_organs)]

    elif img_type == "cone":
        cone_organs = ["kidney", "liver"]
        df = df[df["organ"].isin(cone_organs)]

    ## Getting the list of image paths ##
    high_quality_images_paths = df["hq"].tolist()
    low_quality_images_paths = df["lq"].tolist()

    print(f"Loading the {split} images to memory...")
    ## Getting the list of images in PIL ##
    high_quality_images = [
        Image.open(os.path.join(root_img_path, path))
        for path in high_quality_images_paths
    ]

    low_quality_images = [
        Image.open(os.path.join(root_img_path, path))
        for path in low_quality_images_paths
    ]

    print(f"Loading done!")

    return high_quality_images, low_quality_images


## Defining Pytorch Dataset ##


class EnhanceDataset(Dataset):
    """Implements the custom pytorch dataset, which
    returns the high quality and low quality images.
    """

    def __init__(
        self,
        csv_path: str,
        root_img_path: str,
        split: str = "train",
        train_split_values: list = [0, 1, 2],
        img_type: str = "normal",
        transform=transforms.Compose([transforms.ToTensor()]),
    ):
        """Constructor.

        Parameters
        ----------
        csv_path : str
            The path to the csv file.

        root_img_path : str
            The path to the root folder containing the images.

        split : str
            Split to get the images from. Can be train, test or val.
            [Options : "train", "test", "val"]
            [Default : "train"]

        train_split_values : list
            List of values in "split" to be considered as train
            split.

        img_type : str
            Type of images to get. Can be normal, cone or all.
            [Options : "normal", "cone", "all"]
            [Default : "normal"]

        transform : torchvision.transforms
            The transforms to be applied to the images.

        Note
        ----
        For the validation dataset, keep the train_split_values
        same as that when you were creating the training dataset.
        """

        super().__init__()

        ## Get the list of images ##
        self.high_quality_images, self.low_quality_images = get_images(
            csv_path=csv_path,
            root_img_path=root_img_path,
            split=split,
            train_split_values=train_split_values,
            img_type=img_type,
        )

        ## Define the transforms ##
        self.transform = transform

    def __getitem__(self, idx):
        """Returns the high quality and low quality images at
        the given index, idx, of the datapoint.

        Parameters
        ----------
        idx : int
            The index of the datapoint.
        """

        ## Get the images ##
        high_quality_image = self.high_quality_images[idx]
        low_quality_image = self.low_quality_images[idx]

        ## Apply the transforms ##
        if self.transform:
            high_quality_image = self.transform(high_quality_image)
            low_quality_image = self.transform(low_quality_image)

        return low_quality_image, high_quality_image

    def __len__(self):
        """Returns the length of the dataset."""

        return len(self.high_quality_images)


## Defining the Pytorch Lightning Data Module ##


class EnhanceDataModule(pl.LightningDataModule):
    """The pytorch lightning data module for the ultrasound enhancement
    dataset."""

    def __init__(
        self,
        csv_path: str,
        root_img_path: str,
        img_type: str = "normal",
        train_split_values: list = [0, 1, 2],
        batch_size: int = 32,
        train_transform=transforms.Compose([transforms.ToTensor()]),
        val_transform=transforms.Compose([transforms.ToTensor()]),
    ):
        """Constructor.

        Parameters
        ----------
        csv_path : str
            The path to the csv file.

        root_img_path : str
            The path to the root folder containing all the
            images.

        img_type : str
            Type of images to get. Can be normal, cone or all.
            [Options : "normal", "cone", "all"]
            [Default : "normal"]

        train_split_values : list
            List of values in "split" column to be considered as
            training values.

        batch_size : int
            The batch size to be used for the dataloaders.

        train_transform : torchvision.transforms
            The transforms to be applied to the training images.

        val_transform : torchvision.transforms
            The transforms to be applied to the validation or test
            images.
        """

        super().__init__()

        ## Saving the parameters ##

        self.csv_path = csv_path
        self.root_img_path = root_img_path
        self.img_type = img_type
        self.train_split_values = train_split_values
        self.batch_size = batch_size

        self.train_transform = train_transform
        self.val_transform = val_transform

    def setup(self, stage: str):
        """Setups the dataset for the given stage.

        Parameters
        ----------
        stage : str
            The stage of the model. It can be "fit" or "test".
        """

        if stage == "fit":
            self.enhance_train_dataset = EnhanceDataset(
                csv_path=self.csv_path,
                root_img_path=self.root_img_path,
                split="train",
                train_split_values=self.train_split_values,
                img_type=self.img_type,
                transform=self.train_transform,
            )
            self.enhance_val_dataset = EnhanceDataset(
                csv_path=self.csv_path,
                root_img_path=self.root_img_path,
                split="val",
                train_split_values=self.train_split_values,
                img_type=self.img_type,
                transform=self.val_transform,
            )

        if stage == "test":
            self.enhance_test_dataset = EnhanceDataset(
                csv_path=self.csv_path,
                root_img_path=self.root_img_path,
                split="test",
                img_type=self.img_type,
                transform=self.val_transform,
            )

    def train_dataloader(self):
        """The dataloader instance when the model is in training."""
        return DataLoader(
            self.enhance_train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        """The dataloader instance when the model is in validation."""
        return DataLoader(
            self.enhance_val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        """The dataloader instance when the model is in testing."""
        return DataLoader(
            self.enhance_test_dataset, batch_size=self.batch_size, shuffle=False
        )