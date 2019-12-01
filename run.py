from wrappers import DatasetLoader, Trainer
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch
import sys
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test', help="Path to image/images to test on")
    parser.add_argument('--train', help="Path to folder with training and validation data")
    parser.add_argument('--gpu', default=True, help="Calculations done by GPU or CPU")
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--save_weights', help="Path to save weights after training",
                        default=r"D:\Desktop\Reserve_NNs\weights_configs\defect_detectors\try_1_resnet_cracks\test.pth")

    arguments = parser.parse_args()

    return arguments


def fine_tuning(image_dataset, data_loaders, dataset_sizes,
                class_names, nb_of_epochs, save_path):

    model = models.resnet18(pretrained=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Replace default fc layer with a new one.
    nb_of_filters = model.fc.in_features
    model.fc = nn.Linear(nb_of_filters, len(class_names))

    # Move model to GPU if available
    model.to(device)

    # Initialize loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    # Make sure that only parameters of final layer are being optimized
    optimizer = optim.SGD(model.fc.parameters(),
                          lr=0.001,
                          momentum=0.9)
    # Initialize scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                step_size=7,
                                                gamma=0.1)
    # Initialize trainer
    trainer = Trainer(model=model,
                      loss_function=loss_function,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      nb_of_epochs=nb_of_epochs,
                      data_loaders=data_loaders,
                      dataset_sizes=dataset_sizes,
                      device=device)
    # Train the model
    fit_model = trainer.train()

    # Save the model trained
    torch.save(fit_model.state_dict(), save_path)
    print("Weights saved to:", save_path)


def main():
    args = parse_args()

    if not any((args.test, args.train)):
        print("Incorrect input")
        sys.exit()

    if args.test:
        # testing
        pass
    else:
        # training
        path_to_data = args.train
        nb_of_epochs = int(args.epoch)
        save_weights = args.save_weights

        if not os.path.isdir(path_to_data):
            print("No data provided")
            sys.exit()

        dataset_manager = DatasetLoader(path_to_data)

        image_dataset, data_loaders, dataset_sizes, class_names =\
                                        dataset_manager.generate_datasets()

        fine_tuning(image_dataset, data_loaders, dataset_sizes,
                    class_names, nb_of_epochs, save_weights)


if __name__ == "__main__":
    main()
