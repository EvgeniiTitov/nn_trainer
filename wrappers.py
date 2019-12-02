import torch
import copy
import time
from torchvision import datasets, transforms
import os
import sys
import matplotlib.pyplot as plt


class Visualizer():

    def visualize_training_results(self, accuracy, loss):

        if len(accuracy) > 0 and len(loss) > 0:
            plt.subplot(1, 2, 1)
            plt.plot(accuracy, '-go')
            plt.title("Accuracy")

            plt.subplot(1, 2, 2)
            plt.plot(loss, '-ro')
            plt.title("Loss")

            plt.show()

        return

    def visualize_training_images(self):
        pass


class DatasetLoader():
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
            phase: torch.utils.data.DataLoader(image_datasets[phase], batch_size=4, shuffle=True)
                                for phase in ["train", "val"]
                       }

        dataset_sizes = {
            phase: len(image_datasets[phase]) for phase in ["train", "val"]
                        }

        class_names = image_datasets["train"].classes

        return image_datasets, data_loaders, dataset_sizes, class_names


class Trainer():
    def __init__(self,
                 model,
                 device="cpu"):

        self.model = model
        self.device = device
        self.best_weights = copy.deepcopy(model.state_dict())
        self.best_accuracy = 0.0
        self.visualiser = Visualizer()

    def train(self,
              epochs,
              image_dataset,
              data_loaders,
              dataset_sizes,
              class_names,
              criterion,
              optimizer,
              scheduler):

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
                if phase == "val" and epoch_accuracy > self.best_accuracy:
                    self.best_accuracy = epoch_accuracy
                    self.best_weights = copy.deepcopy(self.model.state_dict())


        training_time = time.time() - start_time
        print("\nTraining completed in:", training_time, " seconds")
        print("BEST ACCURACY:", self.best_accuracy)

        self.model.load_state_dict(self.best_weights)

        # Visualize performance metrics
        #self.visualiser.visualize_training_results(accuracy=val_accuracy_history,
        #                                           loss=val_loss_history)

        return self.model, val_accuracy_history, val_loss_history
