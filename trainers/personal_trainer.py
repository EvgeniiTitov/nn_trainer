import copy
import time
import torch

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