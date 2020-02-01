# python \main.py --train=your_dataset\craks_dataset --epoch=25 --fine_tuning=1 --optimizer=adam --visualise=1

from utils import DatasetLoader, Visualizer
from trainer import GroupTrainer
import torch
import sys
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument('--train', help="Path to folder with train and validation data ImageFolder format")
    parser.add_argument('--fine_tuning', type=int, default=0,
                        help="1 - do not freeze any layers, train all network. 0 - train only classifier")
    parser.add_argument('--train_models', nargs='+', default=False,
                        help="Provide name of the model(s) which you'd like to train or train all")
    parser.add_argument('--pretrained', type=bool, default=True,
                        help="Train models pretrained on the ImageNet ds or not")

    # Extra train arguments
    parser.add_argument('--gpu', default=True, help="Calculations done by GPU or CPU")
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8, help="Number of images per batch")
    parser.add_argument('--optimizer', default="SGD", help="Choose optimizer SGD or ADAM (not case sensitive)")
    parser.add_argument('--classes', type=int, default=2, help="Number of classes to classify")
    parser.add_argument('--early_stopping', type=int, default=0,
                        help="Number of epochs without any loss reduction on val dataset - Early stopping")
    parser.add_argument('--augmentation', type=int, default=0,
                        help="Augmentation of training data: flips, rotation, colour jitter")

    # Results handling
    parser.add_argument('--draw_metrics', type=int, default=0, help="Visualise train metrics upon completion")
    parser.add_argument('--visualise', type=int, default=1, help="Keep track and draw validation loss and accuracy")
    parser.add_argument('--save_weights', help="Path to save weights after train",
                        default=r"D:\Desktop\system_output\Cracks_Training_Results")

    arguments = parser.parse_args()

    return arguments


def train(
        models,
        fine_tuning,
        training_data,
        number_of_epoch,
        batch_size,
        save_path,
        patience,
        optimizer,
        number_of_classes,
        pretrained,
        augmentation
):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the trainer
    trainer = GroupTrainer(
        models=models,
        path_to_training_data=training_data,
        dataloader=DatasetLoader,
        weights_savepath=save_path,
        number_of_classes=number_of_classes,
        device=device,
        fine_tuning=fine_tuning,
        patience=patience,
        batch=batch_size,
        epochs=number_of_epoch,
        optimizer=optimizer,
        pretrained=pretrained,
        augmentation=augmentation
    )

    training_results = trainer.train_models()

    if args.visualise:
        Visualizer.visualize_models_performance(training_results)


if __name__ == "__main__":
    args = parse_args()

    if not args.train:
        print("ERROR: No train data provided!")

    if os.path.isdir(args.train):
        path_to_training_data = args.train
    else:
        print("ERROR: Provided train data is not a folder!")
        sys.exit()

    number_of_epoch = args.epoch
    assert number_of_epoch > 1, "ERROR: Wrong number of epoch"

    batch_size = args.batch_size
    assert batch_size > 1 and batch_size % 8 == 0, "ERROR: Wrong batch size"

    training_type = args.fine_tuning
    number_of_classes = args.classes
    assert number_of_classes > 0, "ERROR: Wrong number of classes"

    if args.early_stopping:

        assert 0 < args.early_stopping < number_of_epoch,\
                                            "ERROR: Wrong patience number of early stopping"
        patience = args.early_stopping
    else:
        patience = 0

    if not args.optimizer.upper().strip() in ["ADAM", "SGD"]:
        print("ERROR: Wrong optimizer provided")
        sys.exit()

    optimizer = args.optimizer

    if not os.path.exists(args.save_weights):
        try:
            os.mkdir(args.save_weights)
        except:
            raise IsADirectoryError("Failed to create a directory to save training results")

        save_path = args.save_weights
    else:
        save_path = args.save_weights

    models_to_train = [
        "resnet18", "resnet34", "resnet50",
        "alexnet", "inception3", "densenet121",
        "squeezenet1_0", "vgg16", "vgg19"
    ]

    if args.train_models:
        models_to_train = [model for model in args.train_models]

    print("\nTRAINING PARAMETERS:")
    print("Models to train:", ' '.join(models_to_train))
    print("Number of classes:", number_of_classes)
    print("Number of epoch:", number_of_epoch)
    print("Fine tuning:", training_type)
    print("Batch size:", batch_size)
    if patience == 0:
        print("No early stopping")
    else:
        print("Early stopping after:", patience)
    print("Optimizer:", optimizer)
    print("Pretrained:", args.pretrained)
    print("Augmentation:", args.augmentation)

    input_confirmation = input("\nCONFIRMED? Y/N: ")

    if input_confirmation.upper().strip() == "Y":
        train(models=models_to_train,
              fine_tuning=training_type,
              training_data=path_to_training_data,
              number_of_epoch=number_of_epoch,
              save_path=save_path,
              batch_size=batch_size,
              patience=patience,
              optimizer=optimizer,
              number_of_classes=number_of_classes,
              pretrained=args.pretrained,
              augmentation=args.augmentation)
    else:
        print("Try again")
        sys.exit()
