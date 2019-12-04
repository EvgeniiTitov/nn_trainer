import torch
import copy
import time
from torchvision import datasets, transforms
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class Visualizer:

    @staticmethod
    def visualize_training_results(accuracy,
                                   loss):

        if len(accuracy) > 0 and len(loss) > 0:
            plt.subplot(1, 2, 1)
            plt.plot(accuracy, '-go')
            plt.title("Accuracy")

            plt.subplot(1, 2, 2)
            plt.plot(loss, '-ro')
            plt.title("Loss")

            plt.show()

        return

    def model_visualisation(self,
                            model,
                            nb_of_images,
                            data_loaders,
                            device,
                            class_names):

        was_training = model.training
        model.eval()
        images_processed = 0
        figure = plt.figure()

        with torch.no_grad():
            for i, (batch_of_images, labels) in enumerate(data_loaders["val"]):
                batch = batch_of_images.to(device)
                labels = labels.to(device)

                activations = model(batch)
                # classes_predicted: tensor([0, 1, 1, 1], device='cuda:0')
                _, classes_predicted = torch.max(activations, dim=1)

                # batch.size(): torch.Size([4, 3, 224, 224])
                for i in range(batch.size()[0]):
                    images_processed += 1

                    ax = plt.subplot(nb_of_images // 2, 2, images_processed)
                    ax.axis("off")

                    ax.set_title(f"Predicted: {class_names[classes_predicted[i]]}")

                    self.show(batch.cpu().data[i])

                    if images_processed == nb_of_images:
                        model.train(mode=was_training)
                        return

        model.train(mode=was_training)


    def show(self, image, title=None):

        img = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.imshow(img)

        if not title is None:
            plt.title(title)

        plt.show()
        plt.pause(0.001)


class DatasetLoader:
    def __init__(self,
                 data_path):
        self.path_to_images = data_path
        self.data_transforms = self.generate_transformations()

    def generate_transformations(self):

        data_transforms = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                          }

        return data_transforms

    def generate_training_datasets(self):

        image_datasets = {
            phase: datasets.ImageFolder(os.path.join(self.path_to_images, phase),
                                                     self.data_transforms[phase]) for phase in ["train", "val"]
                         }

        data_loaders = {
            phase: torch.utils.data.DataLoader(image_datasets[phase], batch_size=8, shuffle=True)
                                for phase in ["train", "val"]
                       }

        dataset_sizes = {
            phase: len(image_datasets[phase]) for phase in ["train", "val"]
                        }

        class_names = image_datasets["train"].classes

        return image_datasets, data_loaders, dataset_sizes, class_names


class Trainer:
    def __init__(self,
                 model,
                 device="cpu"):

        self.model = model
        self.device = device
        self.best_weights = copy.deepcopy(model.state_dict())
        self.best_valid_accuracy = 0.0
        self.best_valid_loss = float("inf")

    def train(self,
              epochs,
              image_dataset,
              data_loaders,
              dataset_sizes,
              class_names,
              criterion,
              optimizer,
              scheduler):

        # Early stopping condition
        epochs_without_improvements = 0
        patience = 7

        val_accuracy_history, val_loss_history = list(), list()
        start_time = time.time()

        print("\nTraining commenced. Computations on:", self.device)

        for epoch in range(epochs):
            print("-" * 30)
            print(f"{epoch + 1} / {epochs}")
            # Each epoch consists of two phases: training and validation
            for phase in ["train", "val"]:

                # Set up the model in accordance with the phase. During evaluation we cannot
                # tweak its parameters - no gradients get calculated, no backprop, no optim. step
                if phase == "train":
                    self.model.train()
                else:
                    self.model.eval()
                # Keep track of model's performance during the epoch
                running_loss, running_corrects = 0.0, 0

                # Load data in batches. Each phase's got its own training and eval data
                for batch, labels in data_loaders[phase]:
                    # Move batches and classes to GPU for faster computation
                    batch = batch.to(self.device)
                    labels = labels.to(self.device)
                    # Zero gradient values
                    optimizer.zero_grad()

                    # Activation all not frozen gradients during training. For validation we do not
                    # need gradients calculated
                    with torch.set_grad_enabled(phase == "train"):
                        # Get predictions for all images in the batch (raw neuron outcomes)
                        activations = self.model(batch)

                        # Get actual classes predicted (run over all activations and pick the largest
                        # value whose index is essentially the class predicted
                        _, classes_predicted = torch.max(activations, dim=1)

                        # Calculate loss value for the batch
                        loss = criterion(activations, labels)

                        # If training phase perform backpropagation and make a gradient step
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # Keep track (add up) all losses and correct predictions for all batches
                    running_loss += loss.item() * batch.size(0)
                    running_corrects += torch.sum(classes_predicted == labels.data)

                # Increment decayer to decrease the learning rate once every N epoch
                if phase == "train":
                    scheduler.step()

                # Calculate average epoch loss and accuracy for each phase by deleting all
                # loss and accuracy values by the total number of images
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_accuracy = running_corrects.double() / dataset_sizes[phase]

                statistics = "{} Loss: {:.4f} Acc: {:.4f}".format(
                                            phase, epoch_loss, epoch_accuracy)
                print(statistics)

                if phase == "val":
                    val_accuracy_history.append(epoch_accuracy)
                    val_loss_history.append(epoch_loss)

                # Check if we got the best accuracy during validation. Save weights if true
                if phase == "val" and epoch_accuracy > self.best_valid_accuracy:
                    self.best_valid_accuracy = epoch_accuracy
                    self.best_weights = copy.deepcopy(self.model.state_dict())

                # Loss tracking for early stopping to prevent overfitting
                if phase == "val" and epoch_loss < self.best_valid_loss:
                    self.best_valid_loss = epoch_loss

                elif phase == "val" and epoch_loss > self.best_valid_loss:
                    if epochs_without_improvements >= patience:
                        print("Early stopping. No loss improvements "
                              "on validation dataset after 7 epochs")
                        break
                    else:
                        epochs_without_improvements += 1

            # Move to the next epoch unless early stopping and we're breaking out
            else:
                continue
            # Break if early stepping and we broke out from the nested loop
            break

        training_time = time.time() - start_time
        print("\nTraining completed in: {:.1f} seconds".format(training_time))

        self.model.load_state_dict(self.best_weights)

        return self.model, val_accuracy_history, val_loss_history


class Tester:

    def __init__(self, model):
        self.model = model
        self.transformator = self.generate_transformations()

    @staticmethod
    def generate_transformations():
        """
        Testing image(s) need to be preprocessed first
        :return:
        """

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                                 )
                                       ])

        return transform

    def preprocess_image(self, path_to_image):

        image = Image.open(path_to_image)
        # Transform image
        image_transformed = self.transformator(image)
        # Unsqueeze image since torch accepts 4D tensors only:
        # batch, channel, height, width
        batch_transformed = torch.unsqueeze(image_transformed, 0)

        return batch_transformed

    def predict(self, batch):

        # perform predictions on the batch provided
        activations = self.model(batch)
        # run over activations, pick index of the largest activation
        _, classes_predicted = torch.max(activations, dim=1)
        # convert activations into percents
        percents = torch.nn.functional.softmax(activations, dim=1)[0]
        # .item() allows to get value from: tensor([0.9006], grad_fn=<IndexBackward>)

        return classes_predicted.item(), percents[classes_predicted].item()

    @staticmethod
    def validation(model,
                   data_loaders,
                   device):

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

        return 100 * correct / total
