import os
import random

import medmnist
import numpy as np
import matplotlib.pyplot as plt
from medmnist import INFO, TissueMNIST
import torchvision.transforms as transforms

NUM_EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 0.001

DIRECTORY_NAME = "TissueMNIST_Dataset"
DATASET_NAME = "tissuemnist"

TRAIN_DIR = os.path.join(DIRECTORY_NAME, "train")
TEST_DIR = os.path.join(DIRECTORY_NAME, "test")
VAL_DIR = os.path.join(DIRECTORY_NAME, "val")

random.seed()  # Seeds from current system time by default

info = INFO[DATASET_NAME]
task = info['task']
num_channels = info['n_channels']
num_classes = len(info['label'])

def print_dataset_info() -> None:
    """
    Prints information about the dataset.

    Args:
        (None)
    Returns:
        (None)
    """
    print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")
    print(f"classes: {num_classes}, channels: {num_channels}, task: {task}")


def get_dataclass_and_transforms():
    """
    Returns a DataClass object, used to download dataset, and Transforms, to apply data augmentation.

    Args:
        (None)
    Returns:
        A tuple consisting of DataClass and Transforms.
    """
    DataClass = getattr(medmnist, info['python_class'])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    return DataClass, data_transform

def create_directories() -> bool:
    """
    Creates directories needed for the TissueMNIST dataset subsets - train, val, test.

    Args:
        (None)
    Returns:
        (bool): True if directories were created.
    """
    if not os.path.exists(DIRECTORY_NAME):
        os.makedirs(DIRECTORY_NAME)
        os.makedirs(TRAIN_DIR)
        os.makedirs(VAL_DIR)
        os.makedirs(TEST_DIR)
        print(f"\nCreated directories:\n{DIRECTORY_NAME}, {TRAIN_DIR}, {VAL_DIR}, {TEST_DIR}\n")
        return True
    else:
        print(f"\nDirectory {DIRECTORY_NAME} already exists. Using existing directory and contents within.\n")
        return False

def view_dataset_contents(dataset:TissueMNIST):
    """
    For the passed dataset, takes random image of index 1-10 and displays its image and label.

    Args:
        (None)
    Returns:
        (None)
    """
    idx = random.randint(1, 10)
    dataset_label_mapping = info['label']

    get_label_as_idx = dataset.labels[idx][0]
    label_name = dataset_label_mapping[str(get_label_as_idx)]       # Because dataset_label_mapping is a dict (str, str)

    plt.imshow(dataset.imgs[idx])
    plt.title(f"Label: {label_name}")

    plt.axis('off')
    plt.show()