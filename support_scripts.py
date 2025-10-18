import argparse
import os
import random
from datetime import datetime
import numpy as np
from PIL import Image

import optuna
from tqdm import tqdm
import pandas as pd
import medmnist
import matplotlib.pyplot as plt
from medmnist import INFO, TissueMNIST
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import ResNet
from torchmetrics.classification import MulticlassAccuracy

NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.004076078213725125

DIRECTORY_NAME = "TissueMNIST_Dataset"
DATASET_NAME = "tissuemnist"
EXPORT_DIRECTORY = "exported_models"

RESULTS_CSV = "optuna_results.csv"

TRAIN_DIR = os.path.join(DIRECTORY_NAME, "train")
TEST_DIR = os.path.join(DIRECTORY_NAME, "test")
VAL_DIR = os.path.join(DIRECTORY_NAME, "val")

random.seed()  # Seeds from current system time by default

info = INFO[DATASET_NAME]
task = info['task']
num_channels = info['n_channels']
num_classes = len(info['label'])

device = "cuda" if torch.cuda.is_available() else "cpu"

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


def get_dataclass_and_transforms(model_name:str):
    """
    Returns a DataClass object, used to download dataset, and Transforms, to apply data augmentation.

    Args:
        model_name (str): Model used, passed through main argument. If `customnet`, then shape (1,H,W) is to be used.
        Otherwise, use (3,H,W) shape.
    Returns:
        A tuple consisting of DataClass and Transforms.
    """
    DataClass = getattr(medmnist, info['python_class'])

    if model_name == "customnet":
        data_transform = transforms.Compose([
            transforms.ToTensor(),  # (1,H,W)
            transforms.Normalize(mean=[.5], std=[.5])
        ])
    else:
        data_transform = transforms.Compose([
            transforms.ToTensor(),                           # (1,H,W)
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # (3,H,W)
            transforms.Normalize(mean=[.5], std=[.5])
        ])

    return DataClass, data_transform


def extract_npz_files():
    print("Extracting npz files...")
    extract_single_npz(f"./{TRAIN_DIR}", "train")
    extract_single_npz(f"./{VAL_DIR}", "val")
    extract_single_npz(f"./{TEST_DIR}", "test")
    print("Extraction complete.")


def extract_single_npz(dataset_path:str, subset_type:str):
    print(f"Extracting {dataset_path}")

    archive_path = os.path.join(dataset_path, "tissuemnist_224.npz")
    if not os.path.exists(archive_path):
        raise FileNotFoundError(f"Expected archive not found: {archive_path}")

    subset_keys = {
        "train": ("train_images", "train_labels"),
        "val": ("val_images", "val_labels"),
        "test": ("test_images", "test_labels"),
    }

    try:
        image_key, label_key = subset_keys[subset_type]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported subset '{subset_type}'. Expected one of: {', '.join(subset_keys)}"
        ) from exc

    save_dir = os.path.join("./tissueMNIST_dataset_extracted", subset_type)
    os.makedirs(save_dir, exist_ok=True)

    label_records = []

    with np.load(archive_path, mmap_mode="r", allow_pickle=False) as data:
        images = data[image_key]
        labels = data[label_key]

        if len(images) != len(labels):
            raise RuntimeError(
                f"Mismatched image/label counts in '{archive_path}' for subset '{subset_type}': "
                f"{len(images)} images vs {len(labels)} labels"
            )

        for index, (img, label) in enumerate(tqdm(zip(images, labels), total=len(images))):
            if isinstance(label, np.ndarray):
                label_id = int(label.item())
            else:
                label_id = int(label)
            label_records.append({"filename": f"{index}.png", "label": label_id})

            label_dir = os.path.join(save_dir, str(label_id))
            os.makedirs(label_dir, exist_ok=True)

            img_uint8 = (img * 255).astype(np.uint8)
            Image.fromarray(img_uint8).save(os.path.join(label_dir, f"{index}.png"))

    labels_csv = os.path.join(save_dir, "labels.csv")
    pd.DataFrame(label_records).to_csv(labels_csv, index=False)

    print(f"Finished extracting {dataset_path}. Labels saved to {labels_csv}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Utility helpers for the TissueMNIST experiments. Use the available "
            "flags to trigger standalone maintenance jobs such as extracting "
            "the 224x224 NPZ archives into image folders."
        )
    )
    parser.add_argument(
        "--extract-npz",
        action="store_true",
        help=(
            "Extract the 224x224 TissueMNIST NPZ archives located inside the "
            "train/val/test directories under TissueMNIST_Dataset."
        ),
    )
    return parser

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


def perform_inference(model : ResNet, dataloader : torch.utils.data.DataLoader) -> None:
    """
    Performs the evaluation of the `model` on data from `dataloader`.

    Args:
        model (ResNet): Model to be evaluated.
        dataloader (torch.utils.data.DataLoader): Dataloader on which to perform evaluation.
    Returns:
        (None)
    """

    pbar = tqdm(dataloader, desc="Evaluating", unit="batch")

    metric = MulticlassAccuracy(num_classes=num_classes)
    model.eval()

    with torch.inference_mode():
        for input_image, target_label in pbar:
            predicted_output = model(input_image)       # Gets output into shape (batch, num_classes)

            # Perform preprocessing on the model's output
            predicted_probabilities = predicted_output.softmax(dim=1)      # Turn logits -> predictions ; Take dim=1 to target num_classes in shape
            #predicted_class = predicted_probabilities.argmax(dim=1)

            target_label = target_label.squeeze()
            metric.update(predicted_probabilities, target_label)

            # Check metrics update within the batch
            acc = metric.compute().item()
            pbar.set_postfix({"Accuracy": f"{acc:.3f}"})

        acc = metric.compute()
        print(f"Accuracy: {acc:.4f}")


def train_model_per_batch(model : ResNet, dataloader : torch.utils.data.DataLoader, loss_fn : torch.nn.Module,
                optimizer: torch.optim.Optimizer) -> None:
    """
    Performs the training of the `model` on `dataloader`. The training within this function is performed per one epoch,
    in a set number of batches. The function tracks the training loss and accuracy.

    Args:
        model (ResNet): Model to be trained.
        dataloader (torch.utils.data.DataLoader): Dataloader on which to perform training.
        loss_fn (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.

    Returns:
        (None)
    """

    pbar = tqdm(dataloader, desc="Training", unit="batch")

    metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    metric.reset()  # Reset at start of each epoch

    total_samples = 0
    train_loss = 0.0

    model.to(device)
    model.train()

    for input_image, target_label in pbar:
        input_image, target_label = input_image.to(device), target_label.to(device)

        predicted_output = model(input_image)
        loss = loss_fn(predicted_output, target_label.squeeze())

        batch_size = input_image.size(0)
        total_samples += batch_size

        train_loss += loss.item() * batch_size
        metric.update(predicted_output, target_label.squeeze())     # Train accuracy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Live tqdm update
        pbar.set_postfix({
            "Train loss": f"{train_loss / total_samples:.4f}",
            "Train accuracy": f"{metric.compute().item():.4f}"
        })


def evaluate_model_per_batch(model : ResNet, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module):
    """
    Performs the evaluation on the `model` on `dataloader`. The evaluation within this function is performed per one epoch,
    in a set number of batches. The function tracks the test loss and accuracy. This function is being called after
    one training iteration has completed, to check the quality of the training iteration.

    Args:
        model (ResNet): Model to be evaluated.
        dataloader (torch.utils.data.DataLoader): Dataloader on which to perform evaluation.
        loss_fn (torch.nn.Module): Loss function.

    Returns:
        (None)
    """
    pbar = tqdm(dataloader, desc="Training validation", unit="batch")

    metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    metric.reset()  # Reset at start of each epoch

    total_samples = 0
    test_loss = 0.0

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for input_image, target_label in pbar:
            input_image, target_label = input_image.to(device), target_label.to(device)

            test_prediction_output = model(input_image)
            loss = loss_fn(test_prediction_output, target_label.squeeze())

            batch_size = input_image.size(0)
            total_samples += batch_size

            test_loss += loss.item() * batch_size
            metric.update(test_prediction_output, target_label.squeeze())  # Test accuracy

            # Live tqdm update
            pbar.set_postfix({
                "Test loss": f"{test_loss / total_samples:.4f}",
                "Test accuracy": f"{metric.compute().item():.4f}"
            })

    final_accuracy = metric.compute().item()
    return final_accuracy

def export_trained_model(model: ResNet) -> None:
    """
    After the training of the model is performed, such model is to be exported, so that later, it can be loaded
    and used for classification immediately. Each time this function is called, the model will have a unique name,
    as each name is generated on the current timestamp when the function was called.

    Args:
        model (ResNet): Trained model to be exported.

    Returns:
        (None)
    """
    os.makedirs(EXPORT_DIRECTORY, exist_ok=True)

    # Create a filename with current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(EXPORT_DIRECTORY, f"model_{timestamp}.pt")

    # Save the model
    torch.save(model.state_dict(), model_path)

    print(f"Model exported to {model_path}")


def objective(trial, model, train_dataloader, val_dataloader):
    """
    Optuna objective function tells Optuna how to train your model and what metric to optimize. It specifies what ranges
    or options to explore for each hyperparameter. Optuna then uses Bayesian optimization (not brute force) to find
    the most optimal values. It learns which regions of search space look promising.

    Hyperparameters varied here are:
    - Learning rate: Controls how aggressively model updates weights.
    - Batch size: Affects gradient noise and convergence stability.
    - Momentum (if SGD is used): Helps SGD escape shallow minima and smooth the updates.
    - Weight Decay: Helps prevent overfitting

    What is not tuned:
    - Number of epochs: Kept constant for fairness (Optuna can't early stop cleanly across trials).
    - Loss function: CrossEntropyLoss is the right choice for ResNet.
    Model architecture: As the used ResNet architecture is being improved.

    Args:
        trial:
    Returns:
        Must return a single scalar (usually validation accuracy or validation loss).
    """
    # --- Hyperparameters to tune ---
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    momentum = trial.suggest_float("momentum", 0.7, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    num_epochs = 3

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_fn = nn.CrossEntropyLoss()

    model.to(device)

    print(f"\nRunning trial {trial.number} with params: {trial.params}")

    # --- Training loop ---
    for epoch in range(num_epochs):
        train_model_per_batch(model, train_dataloader, loss_fn, optimizer)
        val_acc = evaluate_model_per_batch(model, val_dataloader, loss_fn)

        # Report progress to Optuna (for pruning)
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()


    # --- Logging results ---
    print(f"Trial {trial.number} finished: accuracy={val_acc:.4f}, params={trial.params}")

    # Save results to CSV
    results_data = {
        "trial": trial.number,
        "accuracy": val_acc,
        "lr": lr,
        "batch_size": batch_size,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "optimizer": optimizer_name
    }

    # Create or append to results file
    if not os.path.exists(RESULTS_CSV):
        pd.DataFrame([results_data]).to_csv(RESULTS_CSV, index=False)
    else:
        df_existing = pd.read_csv(RESULTS_CSV)
        df_new = pd.concat([df_existing, pd.DataFrame([results_data])], ignore_index=True)
        df_new.to_csv(RESULTS_CSV, index=False)

    return val_acc  # Optuna maximizes this


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.extract_npz:
        extract_npz_files()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
