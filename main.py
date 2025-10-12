import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.models import resnet101, ResNet101_Weights
from support_scripts import create_directories, TRAIN_DIR, TEST_DIR, VAL_DIR, print_dataset_info, \
    get_dataclass_and_transforms, BATCH_SIZE, view_dataset_contents, LEARNING_RATE, perform_inference, num_classes, \
    NUM_EPOCHS, train_model_per_batch, evaluate_model_per_batch

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
    weights = ResNet101_Weights.DEFAULT
    model = resnet101(weights=weights)
    # Replace final layer (Default ResNet101 outputs 1000 logits (prediction classes) per sample -- we need to set it to number of classes of the dataset)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.eval()

    # Perform the classification on the test dataset --- using the original, untrained model
    perform_inference(model, test_dataloader)


    # Train the model, to improve the performance

    # Set the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # TODO: You can try Adam optimizer

    # Perform the training of the model
    pbar = tqdm(range(NUM_EPOCHS), desc="Training the model", unit="epoch")

    for epoch in pbar:
        train_model_per_batch(model, train_dataloader, loss_function, optimizer)
        evaluate_model_per_batch(model, val_dataloader, loss_function)

    # Evaluate the trained model
    perform_inference(model, test_dataloader)