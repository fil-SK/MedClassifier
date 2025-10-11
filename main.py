from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
from medmnist import INFO, Evaluator
import support_scripts
from support_scripts import create_directories, TRAIN_DIR, TEST_DIR, VAL_DIR, print_dataset_info, \
    get_dataclass_and_transforms, BATCH_SIZE

if __name__ == '__main__':
    # Print info about the dataset
    print_dataset_info()

    # Get dataclass and transform
    DataClass, data_transform = get_dataclass_and_transforms()

    # Create dataset directories
    dataset_not_exists = create_directories()
    to_download_dataset = dataset_not_exists

    # Download dataset
    train_dataset = DataClass(split='train', transform=data_transform, download=to_download_dataset, root=f"./{TRAIN_DIR}")
    val_dataset = DataClass(split='val', transform=data_transform, download=to_download_dataset, root=f"./{VAL_DIR}")
    test_dataset = DataClass(split='test', transform=data_transform, download=to_download_dataset, root=f"./{TEST_DIR}")

    # Info on used dataset
    print(train_dataset)
    # print(f"train len: {len(train_dataset)}, val len: {len(val_dataset)}, test len: {len(test_dataset)}")

    # Encapsulate different sub-datasets into DataLoaders
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Visualize the data - a square montage of 10x10 = 100 images from the dataset
    montage_img = train_dataset.montage(length=10)
    plt.imshow(montage_img)
    plt.axis('off')
    plt.show()