from utils import DatasetLoader, Visualizer
from trainers import GroupTrainer, QuantizedTrainer, IndividualTrainer
from testers import Tester
import torch
import sys
import os
import argparse


#TODO:  Add args for scheduler, show batch before training

def parse_args():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument('--train_data',
                        help="Path to folder with train and validation data in the ImageFolder format")
    parser.add_argument('--training_type', type=str, default='group',
                        help="Training type: individual, group, quantized")
    parser.add_argument('--fine_tuning', type=int, default=0,
                        help="1 - do not freeze any layers, train all network. 0 - train only classifier")
    parser.add_argument('--train_models', nargs='+', default=False,
                        help="Provide name(s) of the model(s) to train or leave blank and train all")
    parser.add_argument('--pretrained', type=int, default=1,
                        help="Train models pretrained on the ImageNet DS or not")
    parser.add_argument('--lr', default=0.001, help="Learning rate")
    parser.add_argument('--gpu', default=True, help="Calculations done by GPU or CPU")
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8, help="Number of images per batch")
    parser.add_argument('--optimizer', default="SGD", help="Choose optimizer SGD or ADAM (not case sensitive)")
    parser.add_argument('--classes', type=int, default=2, help="Number of classes to classify")
    parser.add_argument('--early_stopping', type=int, default=5,
                        help="Number of epochs without any loss reduction on val dataset - Early stopping")
    parser.add_argument('--augmentation', type=int, default=1,
                        help="Augmentation of training data: flips, rotation, colour jitter")

    # Results handling
    parser.add_argument('--draw_metrics', type=int, default=1, help="Visualise train metrics upon completion")
    parser.add_argument('--visualise', type=int, default=1, help="Keep track and draw validation loss and accuracy")
    parser.add_argument('--save_weights', help="Path to save weights after train",
                        default=r"D:\Desktop\system_output\dumper_training")
    arguments = parser.parse_args()

    return arguments


def train(
        models_to_train,
        fine_tuning,
        training_data,
        number_of_epoch,
        batch_size,
        save_path,
        patience,
        optimizer,
        number_of_classes,
        pretrained,
        augmentation,
        lr,
        training_type
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if training_type == "group":
        trainer = GroupTrainer(
            path_to_training_data=training_data,
            weights_savepath=save_path,
            number_of_classes=number_of_classes,
            device=device,
            fine_tuning=fine_tuning,
            pretrained=pretrained,
            augmentation=augmentation,
            early_stopping=patience,
            dataset_loader=DatasetLoader
        )
        trainer.train_models(
            models_to_train=models_to_train,
            epochs=number_of_epoch,
            batch=batch_size,
            optimizer_name=optimizer,
            scheduler_required=False,
            lr=lr,
            visualise_batch=True
        )

    elif training_type == "individual":
        #trainer = IndividualTrainer()
        raise NotImplementedError

    elif training_type == "quantized":
        #trainer = QuantizedTrainer()
        raise NotImplementedError


if __name__ == "__main__":
    args = parse_args()

    assert args.train_data and os.path.isdir(args.train_data), "No or incorrect training data provided!"
    path_to_training_data = args.train_data

    assert isinstance(args.epoch, int) and 0 < args.epoch < 1000, "Wrong number of epoch"
    number_of_epoch = args.epoch

    assert isinstance(args.batch_size, int) and args.batch_size > 0 and args.batch_size % 8 == 0,\
                                                                    "Wrong batch size provided"
    batch_size = args.batch_size

    number_of_classes = args.classes
    assert 0 < number_of_classes < 1000, "ERROR: Wrong number of classes"

    if args.early_stopping:
        assert 0 < args.early_stopping < number_of_epoch,\
                                    "ERROR: Wrong patience number for the early stopping condition"
        patience = args.early_stopping
    else:
        patience = 0

    assert args.optimizer.upper().strip() in ["ADAM", "SGD"], "Wrong optimizer provided. Select ADAM or SGD"
    optimizer = args.optimizer

    if not os.path.exists(args.save_weights):
        try:
            os.mkdir(args.save_weights)
        except Exception as e:
            print(f"Failed to create a folder to store training weights. Error: {e}")
            sys.exit()
        save_path = args.save_weights
    else:
        save_path = args.save_weights

    # Either training all layers - fine tuning, or just the classifier
    fine_tuning = False if not args.fine_tuning else True
    if args.pretrained:
        pretrained = True
    else:
        pretrained = False
        print("\n--> WARNING: Pretrained=False, make sure you're training all layers!")
        # If model is not pretrained, do fine tuning - train all layers
        #fine_tuning = True

    models_to_train = [
        "resnet18", "resnet34", "resnet50", "alexnet", "inception3",
        "densenet121", "squeezenet1_0", "vgg16", "vgg19"
    ]

    if args.train_models:
        models_to_train = [model.lower().strip() for model in args.train_models if\
                                                                    model.lower().strip() in models_to_train]

    assert 0 < args.lr < 1, "Check your learning rate"
    assert args.training_type.lower().strip() in ["group", "quantized", "individual"], \
                                        "Wrong training type. Select either Group, Quantized or Individual"

    print("\n----------TRAINING PARAMETERS----------")
    print("Training type:", args.training_type)
    print("Training on:", path_to_training_data)
    print("Models to train:", ' '.join(models_to_train))
    print("Number of classes:", number_of_classes)
    print("-> Fine tuning:", fine_tuning)
    print("-> Pretrained:", pretrained)
    print("Number of epoch:", number_of_epoch)
    print("Batch size:", batch_size)
    if patience == 0:
        print("No early stopping")
    else:
        print("Early stopping after:", patience)
    print("Optimizer:", optimizer)
    print("Learning rate:", args.lr)
    print("Augmentation:", args.augmentation)
    input_confirmation = input("\nCONFIRMED? Y/N: ")

    if input_confirmation.upper().strip() == "Y":
        train(
            models_to_train=models_to_train,
            fine_tuning=fine_tuning,
            training_data=path_to_training_data,
            number_of_epoch=number_of_epoch,
            save_path=save_path,
            batch_size=batch_size,
            patience=patience,
            optimizer=optimizer,
            number_of_classes=number_of_classes,
            pretrained=pretrained,
            augmentation=args.augmentation,
            lr=args.lr,
            training_type=args.training_type
        )
    else:
        print("Try again with correct settings")
        sys.exit()
