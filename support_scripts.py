import os
import random
from datetime import datetime
from tqdm import tqdm
import medmnist
import matplotlib.pyplot as plt
from medmnist import INFO, TissueMNIST
import torch
import torchvision.transforms as transforms
from torchvision.models import ResNet
from torchmetrics.classification import MulticlassAccuracy

NUM_EPOCHS = 3
BATCH_SIZE = 128
LEARNING_RATE = 0.001

DIRECTORY_NAME = "TissueMNIST_Dataset"
DATASET_NAME = "tissuemnist"
EXPORT_DIRECTORY = "exported_models"

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
        transforms.ToTensor(),                           # (1,H,W)
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # (3,H,W)
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


def evaluate_model_per_batch(model : ResNet, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.Module) -> None:
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