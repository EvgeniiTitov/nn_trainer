from wrappers import DatasetLoader, IndividualTrainer, Visualizer, Tester
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch
import sys
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Testing
    parser.add_argument('--test', help="Path to image/images to test on")
    parser.add_argument('--weights', help="Path to model's parameters to test")

    # Training
    parser.add_argument('--train', help="Path to folder with training and validation data")
    parser.add_argument('--learning_type', type=int,
                        help="0 - feature extraction, 1 - fine tuning")

    # Extra training arguments
    parser.add_argument('--gpu', default=True, help="Calculations done by GPU or CPU")
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--draw_metrics', default=False, help="Visualise training metrics upon completion")
    parser.add_argument('--visualize', default=False, help="TBA")
    parser.add_argument('--save_weights', help="Path to save weights after training",
                        default=r"D:\Desktop\Reserve_NNs\weights_configs\defect_detectors\try_1_resnet_cracks")

    arguments = parser.parse_args()

    return arguments


def load_modify_model(nb_of_classes,
                      freezing=False):

    model = models.resnet18(pretrained=True)

    # In case we want to freeze all model's layers
    if freezing:
        for parameter in model.parameters():
            parameter.requires_grad = False

    # Construct new classifier (requires_grad True by default for new layers)
    nb_of_filters = model.fc.in_features
    model.fc = nn.Linear(nb_of_filters, nb_of_classes)

    return model


def check_GPU():

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(model, weights, images):

    # Load model and set it to evaluation mode
    model.load_state_dict(torch.load(weights))
    model.eval()

    tester = Tester(model=model)

    # AlTERNATIVELY PUT ALL IMAGES IN A BATCH AND PROCESS AT THE SAME TIME
    # ON GPU.
    # OR BUILD TEST DATASET LOADER

    for image_path in images:
        image_name = os.path.split(image_path)[-1]

        image_preprocessed = tester.preprocess_image(image_path)
        class_predicted, accuracy = tester.predict(image_preprocessed)
        # Can save image with its class name or whatever
        print("Image: {}, Class: {}, Acc: {:.3f}".format(image_name, class_predicted,
                                                        accuracy*100))


def training(image_dataset,
             data_loaders,
             dataset_sizes,
             class_names,
             nb_of_epochs,
             save_path,
             training_type=0):

    if training_type == 0:
        # Only a new classifier will be trained. Other layers will be frozen
        print("Only new classifier is getting trained")
        model = load_modify_model(nb_of_classes=len(class_names),
                                  freezing=True)

        optimizer = optim.SGD(model.fc.parameters(),
                              lr=0.001,
                              momentum=0.9)
        save_weights = os.path.join(save_path, "feature_extr.pth")

    else:
        # All layers will be trained
        print("All layers are getting trained")
        model = load_modify_model(nb_of_classes=len(class_names),
                                  freezing=False)

        optimizer = optim.SGD(model.parameters(),
                              lr=0.001,
                              momentum=0.9)
        save_weights = os.path.join(save_path, "fine_tuned.pth")

    device = check_GPU()
    model.to(device)

    loss_function = nn.CrossEntropyLoss()

    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                          step_size=7,
                                          gamma=0.1)

    trainer = IndividualTrainer(model=model,
                                device=device)

    fit_model, accuracy, loss = trainer.train(epochs=nb_of_epochs,
                                              image_dataset=image_dataset,
                                              data_loaders=data_loaders,
                                              dataset_sizes=dataset_sizes,
                                              class_names=class_names,
                                              criterion=loss_function,
                                              optimizer=optimizer,
                                              scheduler=scheduler)

    val_acc = Tester.validation(model=fit_model,
                                data_loaders=data_loaders,
                                device=device)

    print("Accuracy on all validation dataset: {:.2f}".format(val_acc))

    visualiser = Visualizer()
    if args.draw_metrics:
        visualiser.visualize_training_results(accuracy, loss)

    if args.visualize:
        visualiser.model_visualisation(fit_model, 10, data_loaders,
                                       device, class_names)

    torch.save(fit_model.state_dict(), save_weights)
    print("\nWeights saved to:", save_weights)


if __name__ == "__main__":
    args = parse_args()

    if not any((args.test, args.train)):
        print("ERROR: Incorrect input. You need to specify either test or train")
        sys.exit()

    # Testing
    if args.test and args.weights:
        # Assume if we provide a file - a photo to classify
        # If we provide a folder - folder of images to classify

        images_to_test = list()

        if os.path.isfile(args.test):
            images_to_test.append(args.test)

        else:
            for image in os.listdir(args.test):
                images_to_test.append(
                                os.path.join(args.test, image)
                                      )

        path_to_weights = args.weights

        test(model=load_modify_model(nb_of_classes=2),
             weights=path_to_weights,
             images = images_to_test)

    elif args.test and not args.weights:
        print("ERROR: You need to provide model's weights for testing")
        sys.exit()

    # Training
    if args.train:

        path_to_data = args.train

        if not os.path.isdir(path_to_data):
            print("ERROR: No data provided")
            sys.exit()

        nb_of_epochs = int(args.epoch)
        save_weights = args.save_weights

        dataset_manager = DatasetLoader(path_to_data)

        image_dataset, data_loaders, dataset_sizes, class_names = \
            dataset_manager.generate_training_datasets()

        if args.learning_type == 0:
            # Only a new classifier gets trained (feature extraction)
            training(image_dataset, data_loaders, dataset_sizes,
                     class_names, nb_of_epochs, save_weights,
                     training_type=0)

        elif args.learning_type == 1:
            # All layers will be trained (fine tuning)
            training(image_dataset, data_loaders, dataset_sizes,
                     class_names, nb_of_epochs, save_weights,
                     training_type=1)

        else:
            print("ERROR: You need to specify training type")
            sys.exit()
