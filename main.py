# python C:\Users\Evgenii\Desktop\Machine_Learning_NNs\work_related_models\main.py
# --train=D:\Desktop\Programming\ML_NN\DataSets\Cats_Dogs_Smaller\cats_vs_dogs
# --pretrained=0 --fine_tuning=1 --epoch=40 --augmentation=1 --train_models resnet18 --visualise=1
from utils import DatasetLoader, Visualizer
from trainer import GroupTrainer
import torch
import sys
import os
import argparse


# TODO: 1. Find mistakes in logic
#       2. Images are of different sizes + smaller than 224. Are you sure they resized and
#          augmented correctly?
#       3. Check your hyperparameters AND what parameters affect training and in what way
#       4. Do some proper tutorials + read your saved article about transfer learning
#       3. Add training options for 1 model with different parameters
#       4. Fix augmentation (remove cropping, add rotations, flips, colour changes)


def parse_args():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument('--train',
                        help="Path to folder with train and validation data in the ImageFolder format")
    parser.add_argument('--fine_tuning', type=int, default=0,
                        help="1 - do not freeze any layers, train all network. 0 - train only classifier")
    parser.add_argument('--train_models', nargs='+', default=False,
                        help="Provide name(s) of the model(s) to train or leave blank and train all")
    parser.add_argument('--pretrained', type=int, default=1,
                        help="Train models pretrained on the ImageNet DS or not")

    # Extra train arguments
    parser.add_argument('--gpu', default=True, help="Calculations done by GPU or CPU")
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8, help="Number of images per batch")
    parser.add_argument('--optimizer', default="SGD", help="Choose optimizer SGD or ADAM (not case sensitive)")
    parser.add_argument('--classes', type=int, default=2, help="Number of classes to classify")
    parser.add_argument('--early_stopping', type=int, default=10,
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

    if not args.train or not os.path.isdir(args.train):
        print("ERROR: No or incorrect training data provided!")
        sys.exit()
    path_to_training_data = args.train

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

    if not args.optimizer.upper().strip() in ["ADAM", "SGD"]:
        print("ERROR: Wrong optimizer provided. Select either ADAM or SGD")
        sys.exit()
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
        # If model is not pretrained, do fine tuning - train all layers
        fine_tuning = True

    models_to_train = [
        "resnet18", "resnet34", "resnet50",
        "alexnet", "inception3", "densenet121",
        "squeezenet1_0", "vgg16", "vgg19"
    ]

    if args.train_models:
        models_to_train = [model.lower().strip() for model in args.train_models if\
                                                                    model.lower().strip() in models_to_train]

    print("\n----------TRAINING PARAMETERS----------")
    print("Models to train:", ' '.join(models_to_train))
    print("Number of classes:", number_of_classes)
    print("Number of epoch:", number_of_epoch)
    print("Fine tuning:", fine_tuning)
    print("Batch size:", batch_size)
    if patience == 0:
        print("No early stopping")
    else:
        print("Early stopping after:", patience)
    print("Optimizer:", optimizer)
    print("Pretrained:", pretrained)
    print("Augmentation:", args.augmentation)

    input_confirmation = input("\nCONFIRMED? Y/N: ")

    if input_confirmation.upper().strip() == "Y":
        train(
            models=models_to_train,
            fine_tuning=fine_tuning,
            training_data=path_to_training_data,
            number_of_epoch=number_of_epoch,
            save_path=save_path,
            batch_size=batch_size,
            patience=patience,
            optimizer=optimizer,
            number_of_classes=number_of_classes,
            pretrained=pretrained,
            augmentation=args.augmentation
        )
    else:
        print("Try again with correct settings")
        sys.exit()
