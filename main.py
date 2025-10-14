import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
import matplotlib.pyplot as plt
import optuna
from torchvision.models import resnet101, ResNet101_Weights, resnet18, ResNet18_Weights
from support_scripts import create_directories, TRAIN_DIR, TEST_DIR, VAL_DIR, print_dataset_info, \
    get_dataclass_and_transforms, BATCH_SIZE, view_dataset_contents, LEARNING_RATE, perform_inference, num_classes, \
    NUM_EPOCHS, train_model_per_batch, evaluate_model_per_batch, export_trained_model, objective

if __name__ == '__main__':
    # Print info about the dataset
    # print_dataset_info()

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
    # print(train_dataset)
    # print(f"train len: {len(train_dataset)}, val len: {len(val_dataset)}, test len: {len(test_dataset)}")

    # View the contents of the dataset (since its npz file)
    #view_dataset_contents(train_dataset)
    #view_dataset_contents(test_dataset)

    # Encapsulate different sub-datasets into DataLoaders
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Visualize the data - a square montage of 10x10 = 100 images from the dataset
    #montage_img = train_dataset.montage(length=10)
    #plt.imshow(montage_img)
    #plt.title("Visualization of 100 images from the dataset")
    #plt.axis('off')
    #plt.show()


    # Instantiate the model
    #weights = ResNet101_Weights.DEFAULT
    #model = resnet101(weights=weights)
    model = resnet18()
    # Replace final layer (Default ResNet101 outputs 1000 logits (prediction classes) per sample -- we need to set it to number of classes of the dataset)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.eval()

    # Perform the classification on the test dataset --- using the original, untrained model
    perform_inference(model, test_dataloader)


    # Train the model, to improve the performance

    # Set the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.004076078213725125)

    # Perform the training of the model
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        train_model_per_batch(model, train_dataloader, loss_function, optimizer)
        evaluate_model_per_batch(model, val_dataloader, loss_function)

    # Evaluate the trained model
    model.cpu() # During evaluation, move model to CPU
    perform_inference(model, test_dataloader)

    export_trained_model(model)


    # Try modifying the hyperparameters
    #model_tuned = resnet101(weights=weights)
    #model_tuned.fc = nn.Linear(model_tuned.fc.in_features, num_classes)     # Replace final layer

    #model_tuned = resnet18(weights=ResNet18_Weights.DEFAULT)
    #model_tuned.fc = nn.Linear(model_tuned.fc.in_features, num_classes)  # Replace final layer

    #study = optuna.create_study(direction="maximize")
    #study.optimize(lambda trial: objective(trial,model_tuned, train_dataloader,val_dataloader,), n_trials=20)  # try 20 hyperparameter sets
    # objective(trial, model_tuned, train_dataloader, val_dataloader)

    #print("Best trial:")
    #trial = study.best_trial

    # Print results
    #print(f"Accuracy: {trial.value:.4f}")
    #print("Params:")
    #for key, value in trial.params.items():
    #    print(f"{key}: {value}")

    # Save trial results
    #study.trials_dataframe().to_csv("optuna_results.csv", index=False)
