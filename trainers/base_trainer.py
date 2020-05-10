import time
import copy
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import utils, models
import numpy as np
import os


class BaseTrainer:
    """

    """
    def __init__(
            self,
            path_to_training_data,
            weights_savepath,
            number_of_classes,
            device,
            fine_tuning,
            pretrained,
            augmentation,
            early_stopping,
            dataset_loader
    ):
        self.path_to_training_data = path_to_training_data
        self.save_path = weights_savepath
        self.nb_of_classes = number_of_classes
        assert 0 < self.nb_of_classes <= 1000, "Wrong number of classes"
        self.device = device
        self.fine_tuning = fine_tuning
        self.pretrained = pretrained
        self.augmentation = augmentation

        if early_stopping:
            self.early_stopping = True
            self.patience = early_stopping
        else:
            self.early_stopping = False

        self.dataset_loader = dataset_loader

    def train(
            self,
            model,
            model_name,
            loss_function,
            optimizer,
            scheduler,
            dataloaders,
            dataset_sizes,
            epochs,
            is_inception=False
    ) -> tuple:
        """

        :param model:
        :param model_name:
        :param loss_function:
        :param optimizer:
        :param scheduler:
        :param dataset_sizes:
        :param epochs:
        :param lr:
        :param early_stopping:
        :param is_inception:
        :return:
        """
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

        for epoch in range(epochs):
            print("-" * 30)
            print(f"{epoch + 1} / {epochs}")
            # Each training epoch consists of training and validation phase
            for phase in ["train", "val"]:

                if phase == "train":
                    model.train()
                else:
                    # No gradients are calculated, no backprop
                    model.eval()

                running_loss, running_corrects = 0.0, 0

                for batch, labels in dataloaders[phase]:
                    try:
                        batch = batch.to(self.device)
                        labels = labels.to(self.device)
                    except Exception as e:
                        print(f"WARNING! Failed to move data to device: {self.device}")

                    optimizer.zero_grad()
                    # Gradients and backprop with subsequent parameters changes are only
                    # allowed during the training phase.
                    with torch.set_grad_enabled(phase == "train"):
                        # For inseption the loss is the sum of the final output and the
                        # auxiliary output. During testing consider only the final one
                        if is_inception and phase == "train":
                            print("Training the exception model, check")
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

                # Decay LR after N epochs
                if phase == "train" and scheduler:
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_accuracy = running_corrects.double() / dataset_sizes[phase]

                print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_accuracy))

                # For visualization
                if phase == "val":
                    val_accuracy_history.append(epoch_accuracy.item())
                    val_loss_history.append(epoch_loss)

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

        return (
            model,
            (val_accuracy_history,
             val_loss_history,
             best_val_accuracy,
             best_accuracy_epoch,
             early_stopped_on)
        )

    def get_params_to_train(self, model) -> list:
        """
        Collects all model's parameters that require training: .requires_grad = True
        :param model:
        :return:
        """
        parameters_to_train = model.parameters()
        # If not fine tuning, hence only the classifier will be trained
        if not self.fine_tuning:
            parameters_to_train = list()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad == True:
                    parameters_to_train.append(parameter)

        return parameters_to_train

    def freeze_model_layers(self, model):
        """
        Freezes model's parameters so that the gradient doesn't propagate down
        the network
        :param model:
        :return:
        """
        for parameter in model.parameters():
            try:
                parameter.requires_grad = False
            except Exception as e:
                print(f"Failed during freezing layers. Error: {e}")
                raise Exception

        return model

    def reshape_model(self, model):
        """
        Reshapes the model by changing its classifier to the number of classes
        the model needs to predict
        :param model:
        :return:
        """
        try:
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
                # Expects (299, 299) images unlike the rest of the networks work with 224, 224
                number_of_filters_AuX = model.AuxLogits.fc.in_features
                number_of_filters = model.fc.in_features
                model.AuxLogits.fc = nn.Linear(number_of_filters_AuX, self.nb_of_classes)
                model.fc = nn.Linear(number_of_filters, self.nb_of_classes)
            elif model.__class__.__name__ == "DenseNet":
                number_of_filters = model.classifier.in_features
                model.classifier = nn.Linear(number_of_filters, self.nb_of_classes)
            elif model.__class__.__name__ == "SqueezeNet":
                # Entirely different output structure. The output comes from 1x1 conv layer,
                # which is the first layer of the classifier
                model.classifier[1] = nn.Conv2d(
                    512,
                    self.nb_of_classes,
                    kernel_size=(1, 1),
                    stride=(1, 1)
                )
                model.num_classes = self.nb_of_classes

        except Exception as e:
            print(f"Failed during model reshaping. Error: {e}")
            raise Exception

        return model

    def visualise_batch(self, batch):
        """

        :param batch:
        :return:
        """
        images, classes = next(iter(batch))
        output = utils.make_grid(images)
        self.show(output)

    def show(self, image, title=None):
        """

        :param image:
        :param title:
        :return:
        """
        img = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.show()
        plt.pause(0.001)

    def visualize_model(
            self,
            dataloaders,
            classnames,
            model,
            rows=3,
            cols=3
    ):
        """
        https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html
        :param model:
        :param rows:
        :param cols:
        :return:
        """
        was_training = model.training
        model.eval()
        current_row = currect_col = 0
        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*2, rows*2))

        with torch.no_grad():
            for idx, (images, labels) in enumerate(dataloaders["val"]):
                images = images.cpu()
                labels = labels.cpu()
                activations = model(images)
                _, predictions = torch.max(activations, dim=1)

                for jdx in range(images.size()[0]):
                    plt.imshow(images.data[jdx], ax=ax[current_row, currect_col])
                    ax[current_row, currect_col].axis("off")
                    ax[current_row, currect_col].set_title(f"predicted: {classnames[predictions[jdx]]}")

                    currect_col += 1
                    if currect_col >= cols:
                        current_row += 1
                        currect_col = 0

                    if current_row >= rows:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    def save_model(
            self,
            model,
            model_name,
            accuracy,
            optimizer
    ) -> None:
        """

        :param model:
        :param model_name:
        :param accuracy:
        :param optimizer:
        :return:
        """
        weights_name = self.get_name_for_weights(model_name, accuracy, optimizer)
        path_to_weights = os.path.join(self.save_path, weights_name)
        torch.save(model, path_to_weights)
        print("Model successfully saved to:", self.save_path)

    def get_name_for_weights(
            self,
            model_name: str,
            accuracy: float,
            optimizer: str
    ) -> str:
        """

        :param model_name:
        :param accuracy:
        :param optimizer:
        :return:
        """
        is_fine_tuning = 1 if self.fine_tuning else 0
        is_pretrained = 1 if self.pretrained else 0

        name = f"{model_name}_Acc{accuracy}_Ftuned{is_fine_tuning}_" \
               f"Pretrained{is_pretrained}_Optimizer{optimizer}.pth"

        return name

    def print_out_training_results(self, training_results: dict) -> None:
        """

        :param training_results:
        :return:
        """
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

    def visualize_training_results(self, models_performance):
        """

        :param models_performance:
        :return:
        """
        labels = list()
        for model_name, performance_metrics in models_performance.items():
            # Metrics were returned in a tuple. Then appended to defaultdict(list), so [][]
            accuracy = performance_metrics[0][0]
            loss = performance_metrics[0][1]
            early_stopping = performance_metrics[0][2]

            plt.subplot(1, 2, 1)
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.plot(accuracy, linewidth=3)

            plt.subplot(1, 2, 2)
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.plot(loss, linewidth=3)

            labels.append(model_name)

        plt.legend(labels)
        plt.show()

    def initialize_a_model(self, model_name):
        """

        :param model_name:
        :return:
        """
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

        return model

    def create_quantized_model(self, model):
        """

        :param model:
        :return:
        """
        if model.__class__.__name__ != "ResNet18":
            raise NotImplementedError("Method implementation is specific for ResNet18")

        num_features = model.fc.in_features
        # 1. Isolate the feature extractor
        model_features = nn.Sequential(
            model.quant,
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
            model.dequant
        )
        # 2. New classifier
        head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, self.nb_of_classes)
        )
        # 3. Combine new classifier and feature extractor
        new_model = nn.Sequential(
            model_features,
            nn.Flatten(1),
            head
        )
        return new_model