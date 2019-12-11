import torch
import copy
import time
from torchvision import datasets, transforms
import torch.optim as optim
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn as nn
from collections import defaultdict

class Visualizer:

    @staticmethod
    def visualize_models_performance(models_performance):

        labels = list()

        for model_name, performance_metrics in models_performance.items():
            # Metrics were returned in a tuple. Then appended to defaultdict(list), so [][]
            accuracy = performance_metrics[0][0]
            loss = performance_metrics[0][1]
            early_stopping = performance_metrics[0][2]

            plt.subplot(1, 2, 1)
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.plot(accuracy, linewidth=2)

            plt.subplot(1, 2, 2)
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.plot(loss, linewidth=2)

            labels.append(model_name)

        plt.legend(labels)
        plt.show()

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
                        model.train_models(mode=was_training)
                        return

        model.train_models(mode=was_training)


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
                 data_path,
                 input_size=224,
                 batch_size=8):

        self.path_to_images = data_path
        self.input_size = input_size
        self.batch_size = batch_size
        self.data_transforms = self.generate_transformations()

    def generate_transformations(self):

        data_transforms = {
            "train": transforms.Compose([
                transforms.RandomResizedCrop(self.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "val": transforms.Compose([
                transforms.Resize(self.input_size),  # 256 used to be
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                          }

        return data_transforms

    def generate_training_datasets(self):

        image_datasets = {
            phase: datasets.ImageFolder(os.path.join(self.path_to_images, phase),
                                                     self.data_transforms[phase])
                                                            for phase in ["train", "val"]
                         }

        data_loaders = {
            phase: torch.utils.data.DataLoader(image_datasets[phase],
                                               batch_size=self.batch_size,
                                               shuffle=True)
                                for phase in ["train", "val"]
                       }

        dataset_sizes = {
            phase: len(image_datasets[phase]) for phase in ["train", "val"]
                        }

        class_names = image_datasets["train"].classes

        return image_datasets, data_loaders, dataset_sizes, class_names


class IndividualTrainer:
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
        patience = 20

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
                    self.model.train_models()
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


class GroupTrainer:

    def __init__(self,
                 models,
                 path_to_training_data,
                 dataloader,
                 weights_savepath,
                 number_of_classes,
                 device,
                 fine_tuning,
                 patience,
                 batch,
                 epochs):

        # List of models to train
        self.models = models

        self.nb_of_classes = number_of_classes
        self.device = device
        self.batch = batch
        self.epochs = epochs

        if patience:
            self.early_stopping = True
            self.patience = patience
        else:
            self.early_stopping = False

        # Path to save weights for each model trained
        self.save_path = weights_savepath
        # Type of training
        self.fine_tuning = fine_tuning
        # Path to the data to train on (ImageFolder type folder)
        self.training_data = path_to_training_data
        # Dataloader that will preprocess training data for us
        self.dataloader = dataloader

        # If feature extraction, freeze models layers
        if not self.fine_tuning:
            self.freeze_layers()
            print("\nModels layers frozen")

        # Reshape models last layer according to the number of classes
        # getting predicted
        self.reshape_models()
        print("\nModels classifiers reshaped to match N of classes")

    def collect_parameters_toupdate(self, model):

        parameters_to_update = model.parameters()

        if not self.fine_tuning:
            parameters_to_update = list()

            for name, parameter in model.named_parameters():
                if parameter.requires_grad == True:
                    parameters_to_update.append(parameter)

        return parameters_to_update

    def train_models(self):

        models_performance = defaultdict(list)

        # Load data set
        dataset_manager = self.dataloader(data_path=self.training_data,
                                          batch_size=self.batch)

        image_dataset, data_loaders, dataset_sizes, class_names = \
                                    dataset_manager.generate_training_datasets()

        for model, model_name in self.models:

            model.to(self.device)
            parameters_to_train = self.collect_parameters_toupdate(model)
            optimizer = optim.SGD(parameters_to_train,
                                  lr=0.001,
                                  momentum=0.9)
            loss_function = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                  step_size=7,
                                                  gamma=0.1)

            # Inspection's input size is different. Generate new dataset if required
            is_inception = False
            if model_name == "inception3":

                dataset_manager = self.dataloader(data_path=self.training_data,
                                                  input_size=299,
                                                  batch_size=self.batch)

                image_dataset, data_loaders, dataset_sizes, class_names = \
                                        dataset_manager.generate_training_datasets()

                is_inception = True

            # Train the model
            model_fit, performance_metrics = self.training(model=model,
                                                           model_name=model_name,
                                                           loss_function=loss_function,
                                                           optimizer=optimizer,
                                                           scheduler=scheduler,
                                                           data_loaders=data_loaders,
                                                           dataset_sizes=dataset_sizes,
                                                           is_inception=is_inception)

            # Once model's been trained, save its performance metrics
            models_performance[model_name].append(performance_metrics)

            # Save model
            path_to_weights = os.path.join(self.save_path, model_name + '.pth')
            torch.save(model_fit.state_dict(), path_to_weights)

        print("All model's weights saved to:", self.save_path)

        return models_performance

    def training(self,
                 model,
                 model_name,
                 loss_function,
                 optimizer,
                 scheduler,
                 data_loaders,
                 dataset_sizes,
                 is_inception):

        start_time = time.time()

        epochs_without_improvements = 0
        early_stopped_on = (False, 0)

        val_accuracy_history, val_loss_history = list(), list()
        best_val_accuracy, best_val_loss = 0, float("inf")

        best_model_weights = copy.deepcopy(model.state_dict())

        print("\nTraining of {} commenced. Calculations on {}".format(
                                                    model_name, self.device))

        for epoch in range(self.epochs):

            print("-" * 30)
            print(f"{epoch + 1} / {self.epochs}")

            for phase in ["train", "val"]:

                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss, running_corrects = 0.0, 0

                for batch, labels in data_loaders[phase]:
                    batch = batch.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        # For inseption the loss is the sum of the final output and the
                        # auxiliary output. During testing consider only the final one
                        if is_inception and phase == "train":
                            # Run batch thru the net, get activations from both layers
                            activations, aux_activations = model(batch)
                            normal_loss = loss_function(activations, labels)
                            aux_loss = loss_function(aux_activations, labels)
                            loss = normal_loss + aux_loss
                        else:
                            activations = model(batch)
                            loss = loss_function(activations, labels)

                        _, class_predictions = torch.max(activations, dim=1)

                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * batch.size(0)
                    running_corrects += torch.sum(class_predictions == labels.data)

                if phase == "train":
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_accuracy = running_corrects.double() / dataset_sizes[phase]

                print("{} Loss: {:.4f} Acc: {:.4f}".format(
                                    phase, epoch_loss, epoch_accuracy
                                                           ))
                # For visualization
                if phase == "val":
                    val_accuracy_history.append(epoch_accuracy.item())
                    val_loss_history.append(epoch_loss)

                # To save the best performing weights
                if phase == "val" and epoch_accuracy > best_val_accuracy:
                    best_val_accuracy = epoch_accuracy
                    best_model_weights = copy.deepcopy(model.state_dict())

                # Early stopping criteria to not overfit
                if self.early_stopping:
                    if phase == "val" and epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss

                    elif phase == "val" and epoch_loss > best_val_loss:
                        if epochs_without_improvements > self.patience:
                            print(f"\n{model_name}'s training early stopped. No loss "
                                  f"improvements on val in {self.patience} epochs")
                            early_stopped_on = (True, epoch)

                            break
                        else:
                            epochs_without_improvements += 1
            # Move to the next epoch unless early stopping and we're breaking out
            else:
                continue
            # Break if nested loop got broken (early stopping)
            break

        training_time = time.time() - start_time
        print("\nTraining completed in: {:.1f} seconds".format(training_time))

        model.load_state_dict(best_model_weights)

        return model, (val_accuracy_history, val_loss_history, early_stopped_on)

    def freeze_layers(self):
        """
        Takes the models provided and freezes their layers
        :return:
        """
        for model, model_name in self.models:

            for parameter in model.parameters():
                parameter.requires_grad = False

        return

    def reshape_models(self):
        """
        Takes the models provided and changes number of outputs from
        1k to the number of classes getting predicted
        :return:
        """
        for model, model_name in self.models:

            if model.__class__.__name__ == "ResNet":
                number_of_filters = model.fc.in_features
                model.fc = nn.Linear(number_of_filters, self.nb_of_classes)

            elif model.__class__.__name__ == "AlexNet":
                # 6th Dense layer's input size: 4096
                number_of_filters = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(number_of_filters, self.nb_of_classes)

            elif model.__class__.__name__ == "VGG":
                # For both VGGs 16-19 classifiers are the same
                number_of_filters = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(number_of_filters, self.nb_of_classes)

            elif model.__class__.__name__ == "Inception3":
                number_of_filters_AuX = model.AuxLogits.fc.in_features
                number_of_filters = model.fc.in_features

                model.AuxLogits.fc = nn.Linear(number_of_filters_AuX, self.nb_of_classes)
                model.fc = nn.Linear(number_of_filters, self.nb_of_classes)

            else:
                print("ERROR: Invalid model's name")
                sys.exit()

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
