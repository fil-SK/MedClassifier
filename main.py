import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt
import optuna
from torchvision.models import resnet101, ResNet101_Weights, resnet18, ResNet18_Weights
from support_scripts import create_directories, TRAIN_DIR, TEST_DIR, VAL_DIR, print_dataset_info, \
    get_dataclass_and_transforms, BATCH_SIZE, view_dataset_contents, LEARNING_RATE, perform_inference, num_classes, \
    NUM_EPOCHS, train_model_per_batch, evaluate_model_per_batch, export_trained_model, objective, MOMENTUM, WEIGHT_DECAY
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--print-dataset-info", action="store_true", help="Prints info about dataset.")
    parser.add_argument("--visualise-data", action="store_true", help="Displays image + label, as well as montage of 100 images.")
    parser.add_argument("--model", type=str, nargs=1, help="Choose a model to use.")
    parser.add_argument("--pretrained-weights", action="store_true", help="Use PyTorch pretrained weights for this model.")
    parser.add_argument("--evaluate-default-model", action="store_true", help="Evaluate a model with default weights.")
    parser.add_argument("--train-model", action="store_true", help="Train a model.")
    parser.add_argument("--set-optimizer", type=str, nargs=1, help="Choose SGD or Adam optimizer.")
    parser.add_argument("--set-lr", nargs=1, type=float, help="Set learning rate, otherwise 0.001 is used.")
    parser.add_argument("--set-epochs", nargs=1, type=int, help="Set number of epochs to train, otherwise 10 is used.")
    parser.add_argument("--set-batch-size", nargs=1, type=int, help="Set batch size for training.")
    parser.add_argument("--optimize-hyperparams", action="store_true", help="Optimize hyperparameters using Optuna framework.")
    args = parser.parse_args()

    # Default values or from arguments
    lr = args.set_lr[0] if args.set_lr else LEARNING_RATE
    epochs = args.set_epochs[0] if args.set_epochs else NUM_EPOCHS
    batch_sz = args.set_batch_size[0] if args.set_batch_size else BATCH_SIZE

    # Get dataclass and transform
    DataClass, data_transform = get_dataclass_and_transforms()

    # Create dataset directories
    dataset_not_exists = create_directories()
    to_download_dataset = dataset_not_exists

    # Download dataset
    train_dataset = DataClass(split='train', transform=data_transform, download=to_download_dataset, root=f"./{TRAIN_DIR}")
    val_dataset = DataClass(split='val', transform=data_transform, download=to_download_dataset, root=f"./{VAL_DIR}")
    test_dataset = DataClass(split='test', transform=data_transform, download=to_download_dataset, root=f"./{TEST_DIR}")

    # Encapsulate different sub-datasets into DataLoaders
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=batch_sz, shuffle=True)
    val_dataloader = data.DataLoader(dataset=val_dataset, batch_size=batch_sz, shuffle=False)
    test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=batch_sz, shuffle=False)



    # Instantiate the model
    if args.model == "resnet18":
        if args.pretrained_weights:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            model = resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)     # Replace final layer (Default ResNet101 outputs 1000 logits (prediction classes) per sample -- we need to set it to number of classes of the dataset)

    elif args.model == "resnet101":
        if args.pretrained_weights:
            model = resnet101(weights=ResNet101_Weights.DEFAULT)
        else:
            model = resnet101()
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    if args.print_dataset_info:
        # Print info about the dataset
        print_dataset_info()

        # Info on used dataset
        print(train_dataset)
        print(f"train len: {len(train_dataset)}, val len: {len(val_dataset)}, test len: {len(test_dataset)}")

    if args.visualise_data:
        # View the contents of the dataset (since its npz file)
        view_dataset_contents(train_dataset)
        view_dataset_contents(test_dataset)

        # Visualize the data - a square montage of 10x10 = 100 images from the dataset
        montage_img = train_dataset.montage(length=10)
        plt.imshow(montage_img)
        plt.title("Visualization of 100 images from the dataset")
        plt.axis('off')
        plt.show()

    if args.evaluate_default_model:
        # Perform the classification on the test dataset --- using the original, untrained model
        perform_inference(model, test_dataloader)

    if args.train_model:
        # Train the model, to improve the performance

        # Set the loss function and optimizer
        loss_function = nn.CrossEntropyLoss()

        if args.set_optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)
        elif args.set_optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

        # Perform the training of the model
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_model_per_batch(model, train_dataloader, loss_function, optimizer)
            evaluate_model_per_batch(model, val_dataloader, loss_function)

        # Evaluate the trained model
        model.cpu() # During evaluation, move model to CPU
        perform_inference(model, test_dataloader)

        export_trained_model(model)

    if args.optimize_hyperparams:
        # Try modifying the hyperparameters
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, model, train_dataloader,val_dataloader,), n_trials=20)  # try 20 hyperparameter sets

        print("Best trial:")
        trial = study.best_trial

        # Print results
        print(f"Accuracy: {trial.value:.4f}")
        print("Params:")
        for key, value in trial.params.items():
            print(f"{key}: {value}")

        # Save trial results
        study.trials_dataframe().to_csv("optuna_results.csv", index=False)
