# python C:\Users\Evgenii\Desktop\Machine_Learning_NNs\work_related_models\main.py --train=D:\Desktop\Programming\ML_NN\DataSets\ants_vs_bees --epoch=5 --fine_tuning=0

from wrappers import DatasetLoader, Visualizer, GroupTrainer
from torchvision.models import resnet18, resnet34, alexnet, vgg16, vgg19, squeezenet, densenet
from torchvision.models.inception import  inception_v3
import torch
import sys
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument('--train', help="Path to folder with training and validation data ImageFolder format")
    parser.add_argument('--fine_tuning', type=int, default=1,
                        help="1 - do not freeze any layers, train all network. 0 - train only classifier")
    parser.add_argument('--train_models', default=False,
                        help="Provide name of the model(s) which you'd like to train or train all")
    parser.add_argument('--pretrained', default=True,
                        help="Train models pretrained on the ImageNet ds or not")

    # Extra training arguments
    parser.add_argument('--gpu', default=True, help="Calculations done by GPU or CPU")
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8, help="Number of images per batch")
    parser.add_argument('--early_stopping', default=False,
                        help="Number of epochs without any loss reduction on val dataset - Early stopping")

    # Results handling
    parser.add_argument('--draw_metrics', default=False, help="Visualise training metrics upon completion")
    parser.add_argument('--visualise', default=False, help="Keep track and draw validation loss and accuracy")
    parser.add_argument('--save_weights', help="Path to save weights after training",
                        default=r"D:\Desktop\Reserve_NNs\weights_configs\defect_detectors\try_2")

    arguments = parser.parse_args()

    return arguments


def training(models,
             fine_tuning,
             training_data,
             number_of_epoch,
             batch_size,
             save_path,
             patience):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the trainer
    trainer = GroupTrainer(
                        models=models,
                        path_to_training_data=training_data,
                        dataloader=DatasetLoader,
                        weights_savepath=save_path,
                        number_of_classes=2,
                        device=device,
                        fine_tuning=fine_tuning,
                        patience=patience,
                        batch=batch_size,
                        epochs=number_of_epoch
                          )

    models_performance = trainer.train_models()

    if args.visualise:
        Visualizer.visualize_models_performance(models_performance)


if __name__ == "__main__":
    args = parse_args()

    if not args.train:
        print("ERROR: Path to training dataset wasn't provided!")

    if os.path.isdir(args.train):
        path_to_training_data = args.train
    else:
        print("ERROR: Provided folder is not a folder!")
        sys.exit()

    number_of_epoch = args.epoch
    batch_size = args.batch_size
    training_type = args.fine_tuning

    if args.early_stopping:
        patience = int(args.early_stopping)
    else:
        patience = False

    if not os.path.exists(args.save_weights):
        try:
            os.mkdir(args.save_weights)
        except:
            print("ERROR: Failed to create a folder to save weights."
                  "Double check your input.")
            sys.exit()
        save_path = args.save_weights
    else:
        save_path = args.save_weights

    models = [
        (resnet18(pretrained=args.pretrained), "resnet18"),
        (resnet34(pretrained=args.pretrained), "resnet34"),
        (alexnet(pretrained=args.pretrained), "alexnet"),
        (vgg16(pretrained=args.pretrained), "vgg16"),
        (vgg19(pretrained=args.pretrained), "vgg19"),
        (inception_v3(pretrained=args.pretrained), "inception3")
             ]

    if args.train_models:
        model_to_train = args.train_models

        models = [(model, model_name) for model, model_name in models
                                if model_name in model_to_train.lower().strip()]

    training(models=models,
             fine_tuning=training_type,
             training_data=path_to_training_data,
             number_of_epoch=number_of_epoch,
             save_path=save_path,
             batch_size=batch_size,
             patience=patience)
