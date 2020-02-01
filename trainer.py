import torch
import copy
import time
import torch.optim as optim
import os
import sys
import torch.nn as nn
from collections import defaultdict
from tqdm import tqdm
from torchvision import models
from pynvml import *


class GroupTrainer:

    def __init__(
            self,
            models,
            path_to_training_data,
            dataloader,
            weights_savepath,
            number_of_classes,
            device,
            fine_tuning,
            patience,
            batch,
            epochs,
            optimizer,
            pretrained,
            augmentation
    ):

        # List of names of models to train
        self.models = models

        # General train hyper parameters
        self.nb_of_classes = number_of_classes
        self.device = device
        self.batch = batch
        self.epochs = epochs
        self.optimizer = optimizer
        self.pretrained = pretrained

        # Early stopping condition
        if patience:
            self.early_stopping = True
            self.patience = patience
        else:
            self.early_stopping = False

        # Path to save weights for each model trained
        self.save_path = weights_savepath
        # Type of train
        self.fine_tuning = fine_tuning
        # Path to the data to train on (ImageFolder type folder)
        self.training_data = path_to_training_data
        # Dataloader that will preprocess train data for us
        self.dataloader = dataloader
        self.augmentation = augmentation

    def collect_parameters_toupdate(self, model):
        """
        Collects model's parameters with requires_grad = True
        :param model:
        :return:
        """
        parameters_to_update = model.parameters()

        if not self.fine_tuning:
            # If feature extraction - last dense layer only
            parameters_to_update = list()

            for name, parameter in model.named_parameters():
                if parameter.requires_grad == True:
                    parameters_to_update.append(parameter)
        else:
            # Fine tuning - collect all parameters
            for name, parameter in model.named_parameters():
                parameters_to_update.append(parameter)

        return parameters_to_update

    @staticmethod
    def print_out_training_results(training_results):

        print("\nTRAINING RESULTS:")
        for model, performance_result in training_results.items():
            print(
                'Model:{} Best acc: {:.4f} on {} epoch'.format(
                                                        model,
                                                        performance_result[0][2],
                                                        performance_result[0][3]
                                                               )
                 )

        return

    def train_models(self):

        models_performance = defaultdict(list)

        # Load data set to train the models
        dataset_manager = self.dataloader(data_path=self.training_data,
                                          augmentation=self.augmentation,
                                          batch_size=self.batch)

        image_dataset, data_loaders, dataset_sizes, class_names = \
                                    dataset_manager.generate_training_datasets()

        # Initialize GPU usage tracking
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)

        # Main training loop
        for model_name in self.models:

            # Print GPU usage stats
            usage = nvmlDeviceGetMemoryInfo(handle)
            print("\nGPU STATS")
            print("Total memory:", usage.total)
            print("Free memory:", usage.free)
            print("Used memory:", usage.used)

            # Initialize the model
            if model_name == "resnet18":
                model = models.resnet18(pretrained=self.pretrained)
            elif model_name == "resnet34":
                model = models.resnet34(pretrained=self.pretrained)
            elif model_name == "resnet50":
                model = models.resnet50(pretrained=self.pretrained)
            elif model_name == "alexnet":
                model = models.alexnet(pretrained=self.pretrained)
            elif model_name == "inception3":
                model = models.inception_v3(pretrained=self.pretrained)
            elif model_name == "densenet121":
                model = models.densenet121(pretrained=self.pretrained)
            elif model_name == "squeezenet1_0":
                model = models.squeezenet1_0(pretrained=self.pretrained)
            elif model_name == "vgg16":
                model = models.vgg16(pretrained=self.pretrained)
            elif model_name == "vgg19":
                model = models.vgg19(pretrained=self.pretrained)
            else:
                raise NameError(f"Invalid name of the model: {model_name}")

            # Feature extraction? freeze model's layers
            if not self.fine_tuning:
                model = self.freeze_layers(model)
                print(f"{model_name}'s layers frozen")

            # Reshape model depending on the number of classes getting classified
            if self.nb_of_classes != 1000:
                model = self.reshape_model(model)
                print(f"{model_name}'s reshaped")

            # Move model to GPU
            model.to(self.device)

            # Collect model's parameters that need to be optimized during training
            parameters_to_train = self.collect_parameters_toupdate(model)

            # Initialize optimizer to perform gradient descend
            if self.optimizer == "SGD":
                optimizer_ = optim.SGD(params=parameters_to_train,
                                       lr=0.001,
                                       momentum=0.9)
            else:
                optimizer_ = optim.Adam(params=parameters_to_train,
                                        lr=0.001,
                                        betas=(0.9, 0.999))

            # Initialize loss function to calculate error
            loss_function = nn.CrossEntropyLoss()

            # Initialize a schedule to degrade the learning rate further into training
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer_,
                                                  step_size=8,
                                                  gamma=0.1)

            # Inspection's input size is different. Generate new dataset if required
            is_inception = False
            if model_name == "inception3":

                dataset_manager = self.dataloader(data_path=self.training_data,
                                                  augmentation=self.augmentation,
                                                  input_size=299,
                                                  batch_size=self.batch)

                image_dataset, data_loaders, dataset_sizes, class_names = \
                                        dataset_manager.generate_training_datasets()

                is_inception = True

            # Trying to catch out of error error
            try:
                # Train the model
                model_fit, performance_metrics = self.training(model=model,
                                                               model_name=model_name,
                                                               loss_function=loss_function,
                                                               optimizer=optimizer_,
                                                               scheduler=scheduler,
                                                               data_loaders=data_loaders,
                                                               dataset_sizes=dataset_sizes,
                                                               is_inception=is_inception)
            except:
                print("\nTRAINING FAILED ON:", model_name)
                continue

            # Keep track of model's training results
            models_performance[model_name].append(performance_metrics)

            # Generate name to save parameters
            acc = round(max(performance_metrics[0]), 4)
            weights_name = model_name + '_' + str(acc) + '_ftuned_' + str(self.fine_tuning) \
                           + '_' + self.optimizer + '.pth'

            # Save model
            path_to_weights = os.path.join(self.save_path, weights_name)
            torch.save(model_fit.state_dict(), path_to_weights)

            # To prevent memory leaks and play it safe
            del model, model_fit, performance_metrics, parameters_to_train,

            # Empty any cache stored by torch for quick allocation without asking OS for space
            torch.cuda.empty_cache()

        print("All model's weights saved to:", self.save_path)

        self.print_out_training_results(models_performance)

        return models_performance

    def train(
            self,
            model,
            model_name,
            loss_function,
            optimizer,
            scheduler,
            data_loaders,
            dataset_sizes,
            is_inception
    ):

        start_time = time.time()

        # Early stopping condition
        epochs_without_improvements = 0
        early_stopped_on = (False, 0)

        # Tracking train process
        val_accuracy_history, val_loss_history = list(), list()
        best_val_accuracy, best_val_loss = 0, float("inf")
        best_accuracy_epoch = 0


        best_model_weights = copy.deepcopy(model.state_dict())

        print("\nTraining of {} commenced. Calculations on {}".format(
                                                    model_name, self.device))

        for epoch in range(self.epochs):

            print("-" * 30)
            print(f"{epoch + 1} / {self.epochs}")

            # Each training epoch consists of training and validation phase
            for phase in ["train", "val"]:

                if phase == "train":
                    model.train()
                else:
                    # No gradients are calculated, no backprop
                    model.eval()

                running_loss, running_corrects = 0.0, 0

                for batch, labels in data_loaders[phase]:
                    batch = batch.to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()

                    # Gradients and backprop with subsequent parameters changes are only
                    # allowed during the training phase.
                    with torch.set_grad_enabled(phase == "train"):
                        # For inseption the loss is the sum of the final output and the
                        # auxiliary output. During testing consider only the final one
                        if is_inception and phase == "train":
                            # Run batch thru the net, get activations from both layers
                            activations, aux_activations = model(batch)
                            normal_loss = loss_function(activations, labels)
                            aux_loss = loss_function(aux_activations, labels)
                            loss = normal_loss + (0.4 * aux_loss)
                        else:
                            activations = model(batch)
                            loss = loss_function(activations, labels)

                        _, class_predictions = torch.max(activations, dim=1)

                        # Run backprop and optimize values if in training
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * batch.size(0)
                    running_corrects += torch.sum(class_predictions == labels.data)

                # TO CONFIRM
                # if phase == "train":
                #     scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_accuracy = running_corrects.double() / dataset_sizes[phase]

                print("{} Loss: {:.4f} Acc: {:.4f}".format(
                                    phase, epoch_loss, epoch_accuracy
                                                           ))
                # For visualization
                if phase == "val":
                    val_accuracy_history.append(epoch_accuracy.item())
                    val_loss_history.append(epoch_loss.item())

                # To save the best performing weights
                if phase == "val" and epoch_accuracy > best_val_accuracy:
                    best_val_accuracy = epoch_accuracy.item()
                    best_model_weights = copy.deepcopy(model.state_dict())
                    best_accuracy_epoch = epoch

                # Early stopping criteria to prevent overfitting
                if self.early_stopping:
                    if phase == "val" and epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss

                    elif phase == "val" and epoch_loss > best_val_loss:
                        if epochs_without_improvements > self.patience:
                            print(f"\n{model_name}'s train early stopped. No loss "
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

        # Load best model weights
        model.load_state_dict(best_model_weights)

        return model, (val_accuracy_history, val_loss_history,
               best_val_accuracy, best_accuracy_epoch, early_stopped_on)

    def freeze_layers(self, model):
        """
        Receives a model and freezes its layers - gradient wont
        propagate along the layers
        :return: model modified
        """

        for parameter in model.parameters():
            parameter.requires_grad = False

        return model

    def reshape_model(self, model):
        """
        Receives a model and changes the number of outputs from
        1k to the number of classes getting predicted
        :return: reshaped model
        """

        if model.__class__.__name__ == "ResNet":
            number_of_filters = model.fc.in_features
            model.fc = nn.Linear(number_of_filters, self.nb_of_classes)
            print("Resnet successfully reshaped")

        elif model.__class__.__name__ == "AlexNet":
            # 6th Dense layer's input size: 4096
            number_of_filters = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(number_of_filters, self.nb_of_classes)
            print("AlexNet successfully reshaped")

        elif model.__class__.__name__ == "VGG":
            # For both VGGs 16-19 classifiers are the same
            number_of_filters = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(number_of_filters, self.nb_of_classes)
            print("VGG successfully reshaped")

        elif model.__class__.__name__ == "Inception3":
            # Expects (299, 299) images unlike the rest of the networks work with 224, 224
            number_of_filters_AuX = model.AuxLogits.fc.in_features
            number_of_filters = model.fc.in_features

            model.AuxLogits.fc = nn.Linear(number_of_filters_AuX, self.nb_of_classes)
            model.fc = nn.Linear(number_of_filters, self.nb_of_classes)
            print("Inception successfully reshaped")

        elif model.__class__.__name__ == "DenseNet":
            number_of_filters = model.classifier.in_features
            model.classifier = nn.Linear(number_of_filters, self.nb_of_classes)
            print("DenseNet successfully reshaped")

        elif model.__class__.__name__ == "SqueezeNet":
            # Entirely different output structure. The output comes from 1x1 conv layer,
            # which is the first layer of the classifier
            model.classifier[1] = nn.Conv2d(512,
                                            self.nb_of_classes,
                                            kernel_size=(1, 1),
                                            stride=(1, 1))
            model.num_classes = self.nb_of_classes
            print("SqueezeNet successfully reshaped")

        return model


class IndividualTrainer:

    def __init__(
            self,
            model,
            device="cpu"
    ):

        self.model = model
        self.device = device
        self.best_weights = copy.deepcopy(model.state_dict())
        self.best_valid_accuracy = 0.0
        self.best_valid_loss = float("inf")

    def train(
            self,
            epochs,
            image_dataset,
            data_loaders,
            dataset_sizes,
            class_names,
            criterion,
            optimizer,
            scheduler
    ):

        # Early stopping condition
        epochs_without_improvements = 0
        patience = 20

        val_accuracy_history, val_loss_history = list(), list()
        start_time = time.time()

        print("\nTraining commenced. Computations on:", self.device)

        for epoch in range(epochs):
            print("-" * 30)
            print(f"{epoch + 1} / {epochs}")
            # Each epoch consists of two phases: train and validation
            for phase in ["train", "val"]:

                # Set up the model in accordance with the phase. During evaluation we cannot
                # tweak its parameters - no gradients get calculated, no backprop, no optim. step
                if phase == "train":
                    self.model.train_models()
                else:
                    self.model.eval()
                # Keep track of model's performance during the epoch
                running_loss, running_corrects = 0.0, 0

                # Load data in batches. Each phase's got its own train and eval data
                for batch, labels in data_loaders[phase]:
                    # Move batches and classes to GPU for faster computation
                    batch = batch.to(self.device)
                    labels = labels.to(self.device)
                    # Zero gradient values
                    optimizer.zero_grad()

                    # Activation all not frozen gradients during train. For validation we do not
                    # need gradients calculated
                    with torch.set_grad_enabled(phase == "train"):
                        # Get predictions for all images in the batch (raw neuron outcomes)
                        activations = self.model(batch)

                        # Get actual classes predicted (run over all activations and pick the largest
                        # value whose index is essentially the class predicted
                        _, classes_predicted = torch.max(activations, dim=1)

                        # Calculate loss value for the batch
                        loss = criterion(activations, labels)

                        # If train phase perform backpropagation and make a gradient step
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
