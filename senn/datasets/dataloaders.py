import os
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import random
from PIL import ImageDraw, Image


def get_dataloader(config):
    """Dispatcher that calls dataloader function depending on the configs."""
    if config.dataloader.lower() == 'mnist':
        return load_mnist(**config.__dict__)
    elif config.dataloader.lower() == 'fashion-mnist':
        return load_fashion_mnist(**config.__dict__)
    elif config.dataloader.lower() == 'confounded-mnist':
        return load_confounded_mnist(**config.__dict__)
    elif config.dataloader.lower() == 'confounded-fashionmnist':
        return load_confounded_fashionmnist(**config.__dict__)
    elif config.dataloader.lower() == 'compas':
        return load_compas(**config.__dict__)


class ConfoundedDataset(Dataset):
    def __init__(self, dataset, is_train=True, dot_size=3):
        self.dataset = dataset
        self.is_train = is_train
        self.dot_size = dot_size
        self.num_classes = 10
        self.fixed_positions = self._generate_fixed_positions()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Store a fixed mapping of dot positions for the test set
        self.test_dot_positions = {}
        if not self.is_train:
            self._assign_fixed_test_positions()

        if hasattr(self.dataset, "data"):
            self.data = self.dataset.data  # Reference dataset data

    def _generate_fixed_positions(self):
        """Defines unique positions for each class in the 28x28 image grid."""
        positions = {
            0: (2, 2),      
            1: (2, 25),     
            2: (25, 2),     
            3: (25, 25),    
            4: (14, 0),     
            5: (14, 25),    
            6: (2, 14),     
            7: (25, 14),    
            8: (26, 22),    
            9: (20, -1),
        }
        return positions

    def _assign_fixed_test_positions(self):
        """
        Assigns a **fixed** confounder dot position to each test sample.
        Ensures the same sample always gets the same dot placement.
        """
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]  # Retrieve label
            other_classes = list(set(self.fixed_positions.keys()) - {label})
            mapped_class = other_classes[idx % len(other_classes)]  # Deterministic mapping
            self.test_dot_positions[idx] = self.fixed_positions[mapped_class]

    def _add_dot(self, image, label, idx):
        """Adds the confounder dot at a fixed position."""
        draw = ImageDraw.Draw(image)

        if self.is_train:
            position = self.fixed_positions[label]
        else:
            # Use the **stored** test dot position to ensure consistency
            position = self.test_dot_positions[idx]

        draw.ellipse(
            [position[0], position[1], position[0] + self.dot_size, position[1] + self.dot_size],
            fill=255
        )
        return image

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = self._add_dot(image, label, idx)  # Add confounder at fixed test position
        image = self.transform(image)  # Convert PIL image to tensor
        return image, label







def load_confounded_mnist(data_path, batch_size, num_workers=0, valid_size=0.1, **kwargs):
    return load_confounded_dataset(data_path, batch_size, num_workers, valid_size, dataset_type='mnist')

def load_confounded_fashionmnist(data_path, batch_size, num_workers=0, valid_size=0.1, **kwargs):
    return load_confounded_dataset(data_path, batch_size, num_workers, valid_size, dataset_type='fashion-mnist')

def load_confounded_dataset(data_path, batch_size, num_workers=0, valid_size=0.1, dataset_type='mnist'):
    dataset_cls = MNIST if dataset_type == 'mnist' else FashionMNIST

    # REMOVE `transforms.ToPILImage()`, USE `transform=None`
    train_set = dataset_cls(data_path, train=True, download=True, transform=None)
    test_set = dataset_cls(data_path, train=False, download=True, transform=None)
    
    confounded_train_set = ConfoundedDataset(train_set, is_train=True)
    confounded_test_set = ConfoundedDataset(test_set, is_train=False)
    
    train_size = len(confounded_train_set)
    split = int(np.floor(valid_size * train_size))
    indices = list(range(train_size))
    train_sampler = SubsetRandomSampler(indices[split:])
    valid_sampler = SubsetRandomSampler(indices[:split])
    
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers, drop_last=True)
    train_loader = DataLoader(confounded_train_set, sampler=train_sampler, **dataloader_args)
    valid_loader = DataLoader(confounded_train_set, sampler=valid_sampler, **dataloader_args)
    test_loader = DataLoader(confounded_test_set, shuffle=False, **dataloader_args)
    
    return train_loader, valid_loader, test_loader



def load_fashion_mnist(data_path, batch_size, num_workers=0, valid_size=0.1, **kwargs):
    """
    Load Fashion-MNIST data.

    Performs the following preprocessing operations:
        - converting to tensor
        - standard normalization used for Fashion-MNIST

    Parameters
    ----------
    data_path: str
        Location of Fashion-MNIST data.
    batch_size: int
        Batch size.
    num_workers: int
        Number of workers for the PyTorch DataLoaders.
    valid_size : float
        A float between 0.0 and 1.0 for the percent of samples to be used for validation.

    Returns
    -------
    train_loader
        Dataloader for training set.
    valid_loader
        Dataloader for validation set.
    test_loader
        Dataloader for testing set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Mean and std for Fashion-MNIST
    ])

    train_set = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)

    train_size = len(train_set)
    split = int(np.floor(valid_size * train_size))
    indices = list(range(train_size))
    train_sampler = SubsetRandomSampler(indices[split:])
    valid_sampler = SubsetRandomSampler(indices[:split])

    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers, drop_last=True)
    train_loader = DataLoader(train_set, sampler=train_sampler, **dataloader_args)
    valid_loader = DataLoader(train_set, sampler=valid_sampler, **dataloader_args)
    test_loader = DataLoader(test_set, shuffle=False, **dataloader_args)

    return train_loader, valid_loader, test_loader

def load_mnist(data_path, batch_size, num_workers=0, valid_size=0.1, **kwargs):
    """
    Load mnist data.

    Loads mnist dataset and performs the following preprocessing operations:
        - converting to tensor
        - standard mnist normalization so that values are in (0, 1)

    Parameters
    ----------
    data_path: str
        Location of mnist data.
    batch_size: int
        Batch size.
    num_workers: int
        the number of  workers to be used by the Pytorch DataLoaders
    valid_size : float
        a float between 0.0 and 1.0 for the percent of samples to be used for validation

    Returns
    -------
    train_loader
        Dataloader for training set.
    valid_loader
        Dataloader for validation set.
    test_loader
        Dataloader for testing set.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(data_path, train=False, download=True, transform=transform)

    train_size = len(train_set)
    split = int(np.floor(valid_size * train_size))
    indices = list(range(train_size))
    train_sampler = SubsetRandomSampler(indices[split:])
    valid_sampler = SubsetRandomSampler(indices[:split])

    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers, drop_last=True)
    train_loader = DataLoader(train_set, sampler=train_sampler, **dataloader_args)
    valid_loader = DataLoader(train_set, sampler=valid_sampler, **dataloader_args)
    test_loader = DataLoader(test_set, shuffle=False, **dataloader_args)

    return train_loader, valid_loader, test_loader


#  --------------- Compas Dataset  ---------------

class CompasDataset(Dataset):
    def __init__(self, data_path, verbose=True):
        """ProPublica Compas dataset.

        Dataset is read in from preprocessed compas data: `propublica_data_for_fairml.csv`
        from fairml github repo.
        Source url: 'https://github.com/adebayoj/fairml/raw/master/doc/example_notebooks/propublica_data_for_fairml.csv'
        
        Following approach of Alvariz-Melis et al (SENN).
        
        Parameters
        ----------
        data_path : str
            Location of Compas data.
        """
        df = pd.read_csv(data_path)

        # don't know why square root
        df['Number_of_Priors'] = (df['Number_of_Priors'] / df['Number_of_Priors'].max()) ** (1 / 2)
        # get target
        compas_rating = df.score_factor.values  # This is the target?? (-_-)
        df = df.drop('score_factor', axis=1)

        pruned_df, pruned_rating = find_conflicting(df, compas_rating)
        if verbose:
            print('Finish preprocessing data..')

        self.X = pruned_df
        self.y = pruned_rating.astype(float)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return self.X.iloc[idx].values.astype(float), self.y[idx]


def load_compas(data_path='senn/datasets/data/compas/compas.csv', train_percent=0.8, batch_size=200,
                num_workers=0, valid_size=0.1, **kwargs):
    """Return compas dataloaders.
    
    If compas data can not be found, will download preprocessed compas data: `propublica_data_for_fairml.csv`
    from fairml github repo.
    
    Source url: 'https://github.com/adebayoj/fairml/raw/master/doc/example_notebooks/propublica_data_for_fairml.csv'

    Parameters
    ----------
    data_path : str
        Path of compas data.
    train_percent : float
        What percentage of samples should be used as the training set. The rest is used
        for the test set.
    batch_size : int
        Number of samples in minibatches.

    Returns
    -------
    train_loader
        Dataloader for training set.
    valid_loader
        Dataloader for validation set.
    test_loader
        Dataloader for testing set.
    """
    if not os.path.isfile(data_path):
        Path(data_path).parent.mkdir(parents=True, exist_ok=True)
        compas_url = 'https://github.com/adebayoj/fairml/raw/master/doc/example_notebooks/propublica_data_for_fairml.csv'
        download_file(data_path, compas_url)
    dataset = CompasDataset(data_path)

    # Split into training and test
    train_size = int(train_percent * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    indices = list(range(train_size))
    validation_split = int(valid_size * train_size)
    train_sampler = SubsetRandomSampler(indices[validation_split:])
    valid_sampler = SubsetRandomSampler(indices[:validation_split])

    # Dataloaders
    dataloader_args = dict(batch_size=batch_size, num_workers=num_workers, drop_last=True)
    train_loader = DataLoader(train_set, sampler=train_sampler, **dataloader_args)
    valid_loader = DataLoader(train_set, sampler=valid_sampler, **dataloader_args)
    test_loader = DataLoader(test_set, shuffle=False, **dataloader_args)

    return train_loader, valid_loader, test_loader


def find_conflicting(df, labels, consensus_delta=0.2):
    """
    Find examples with same exact feature vector but different label.

    Finds pairs of examples in dataframe that differ only in a few feature values.

    From SENN authors' code.

    Parameters
    ----------
    df : pd.Dataframe
        Containing compas data.
    labels : iterable
        Containing ground truth labels
    consensus_delta : float
        Decision rule parameter.

    Return
    ------
    pruned_df:
        dataframe with `inconsistent samples` removed.
    pruned_lab:
        pruned labels
    """

    def finder(df, row):
        for col in df:
            df = df.loc[(df[col] == row[col]) | (df[col].isnull() & pd.isnull(row[col]))]
        return df

    groups = []
    all_seen = set([])
    full_dups = df.duplicated(keep='first')
    for i in (range(len(df))):
        if full_dups[i] and (i not in all_seen):
            i_dups = finder(df, df.iloc[i])
            groups.append(i_dups.index)
            all_seen.update(i_dups.index)

    pruned_df = []
    pruned_lab = []
    for group in groups:
        scores = np.array([labels[i] for i in group])
        consensus = round(scores.mean())
        for i in group:
            if (abs(scores.mean() - 0.5) < consensus_delta) or labels[i] == consensus:
                # First condition: consensus is close to 50/50, can't consider this "outliers", so keep them all
                pruned_df.append(df.iloc[i])
                pruned_lab.append(labels[i])
    return pd.DataFrame(pruned_df), np.array(pruned_lab)


def download_file(store_path, url):
    """Download a file from `url` and write it to a file `store_path`.

    Parameters
    ----------
    store_path : str
        Data storage location.
    """
    # Download the file from `url` and save it locally under `file_name`
    with urllib.request.urlopen(url) as response, open(store_path, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
