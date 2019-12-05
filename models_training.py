from wrappers import DatasetLoader, IndividualTrainer, Visualizer, Tester, GroupTrainer
from torchvision.models import resnet18, resnet34, alexnet, vgg16, vgg19,\
                               squeezenet, densenet
from torchvision.models.inception import  inception_v3
import torch.nn as nn
import torch.optim as optim
import torch
import sys
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument('--train', help="Path to folder with training and validation data")
    parser.add_argument('--fine_tuning', type=str, default=True,
                        help="True - do not freeze any layers, False - train only classifier")
    parser.add_argument('--train_specific', default=False,
                        help="Provide name of the model which you'd like to train or train all")
    parser.add_argument('--pretrained', default=True,
                        help="Train models pretrained on ImageNet or not")

    # Extra training arguments
    parser.add_argument('--gpu', default=True, help="Calculations done by GPU or CPU")
    parser.add_argument('--epoch', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8, help="Number of images per batch")
    parser.add_argument('--draw_metrics', default=False, help="Visualise training metrics upon completion")
    parser.add_argument('--visualize', default=False, help="TBA")
    parser.add_argument('--save_weights', help="Path to save weights after training",
                        default=r"D:\Desktop\Reserve_NNs\weights_configs\defect_detectors\try_1_resnet_cracks")

    arguments = parser.parse_args()

    return arguments


def training(models,
             fine_tuning,
             training_data,
             number_of_epoch,
             batch_size,
             save_path):

    """

    :param models:
    :param fine_tuning:
    :param training_data:
    :param number_of_epoch:
    :param batch_size:
    :param save_path:
    :return:
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize the trainer
    trainer = GroupTrainer(
                        models=models,
                        path_to_training_data=training_data,
                        dataloader=DatasetLoader,
                        fine_tuning=fine_tuning,
                        weights_savepath=save_path,
                        number_of_classes=2,
                        device=device
                          )


    models_performance = trainer.train_models(
                                    epochs=number_of_epoch,
                                    batch=batch_size,
                                    criterion=,
                                    optimizer=,
                                    scheduler=,
                                      )





if __name__ == "__main__":
    args = parse_args()

    if not all((args.train, args.training_type)):
        print("ERROR: Incorrect input")

    if os.path.isdir(args.train):
        path_to_training_data = args.train
    else:
        print("ERROR: Provided folder is not a folder!")
        sys.exit()


    number_of_epoch = int(args.epoch)
    batch_size = int(args.batch_size)
    training_type = args.fine_tuning
    save_path = args.save_weights

    models = [
        resnet18(pretrained=args.pretrained),
        resnet34(pretrained=args.pretrained),
        alexnet(pretrained=args.pretrained),
        vgg16(pretrained=args.pretrained),
        vgg19(pretrained=args.pretrained),
        inception_v3(pretrained=args.pretrained)
    ]

    # NOW IT GIVES BOTH VGG for input VGG, same for ResNet
    if args.train_specific:
        model_to_train = args.train_specific

        models = [model for model in models if model.__class__.__name__.lower()==\
                                                            model_to_train.lower().strip()]

    training(models=models,
             fine_tuning=training_type,
             training_data=path_to_training_data,
             number_of_epoch=number_of_epoch,
             save_path=save_path,
             batch_size=batch_size)
