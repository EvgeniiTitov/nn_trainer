from wrappers import DatasetLoader, Trainer, Visualizer
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
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--draw_metrics', default=False, help="Visualise training metrics upon completion")
    parser.add_argument('--visualize', default=False, help="TBA")
    parser.add_argument('--save_weights', help="Path to save weights after training",
                        default=r"D:\Desktop\Reserve_NNs\weights_configs\defect_detectors\try_1_resnet_cracks\test.pth")

    arguments = parser.parse_args()

    return arguments


def test_model(model, data_loaders, dataset_sizes, device):
    correct, total = 0, 0

    with torch.no_grad():
        for batch, labels in data_loaders["val"]:
            batch_of_images = batch.to(device)
            labels = labels.to(device)

            batch_activations = model(batch_of_images)
            # batch_predictions: tensor([0, 1, 1, 1], device='cuda:0')
            _, batch_predictions = torch.max(batch_activations.data, dim=1)

            total += labels.size(0)
            correct += (batch_predictions == labels).sum().item()

    print("\nAccuracy on {} valid images: {:.4f}".format(dataset_sizes["val"], 100*correct/total))


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
                      device=device)
    # Train the model
    fit_model, accuracy, loss = trainer.train(epochs=nb_of_epochs,
                                              image_dataset=image_dataset,
                                              data_loaders=data_loaders,
                                              dataset_sizes=dataset_sizes,
                                              class_names=class_names,
                                              criterion=loss_function,
                                              optimizer=optimizer,
                                              scheduler=scheduler)

    # Test model's performance on all validation images
    test_model(fit_model, data_loaders, dataset_sizes, device)

    visualiser = Visualizer()
    if args.draw_metrics:
        visualiser.visualize_training_results(accuracy, loss)

    if args.visualize:
        visualiser.model_visualisation(fit_model, 10, data_loaders,
                                       device, class_names)

    # Save the model trained
    torch.save(fit_model.state_dict(), save_path)
    print("\nWeights saved to:", save_path)


if __name__ == "__main__":
    args = parse_args()

    if not any((args.test, args.train)):
        print("Incorrect input. You need to specify either test or train")
        sys.exit()

    if args.test:
        path_to_weights = args.test
        #test_model(path_to_weights)

    else:
        # training
        path_to_data = args.train
        nb_of_epochs = int(args.epoch)
        save_weights = args.save_weights

        if not os.path.isdir(path_to_data):
            print("No data provided")
            sys.exit()

        if not os.path.exists(save_weights):
            os.mkdir(save_weights)

        dataset_manager = DatasetLoader(path_to_data)

        image_dataset, data_loaders, dataset_sizes, class_names = \
            dataset_manager.generate_training_datasets()

        fine_tuning(image_dataset, data_loaders, dataset_sizes,
                    class_names, nb_of_epochs, save_weights)
