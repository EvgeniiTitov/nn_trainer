import torch
import torch.optim as optim
import torch.nn as nn
from collections import defaultdict
from torchvision import models, utils
from trainers.base_trainer import BaseTrainer


class GroupTrainer(BaseTrainer):
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
        super().__init__(
            path_to_training_data,
            weights_savepath,
            number_of_classes,
            device,
            fine_tuning,
            pretrained,
            augmentation,
            early_stopping,
            dataset_loader
        )

    def train_models(
            self,
            models_to_train,
            epochs,
            batch,
            optimizer_name,
            scheduler_required,
            lr,
            visualise_batch
    ):
        """

        :param models_to_train:
        :param epochs:
        :param batch:
        :param optimizer_name:
        :param scheduler_required:
        :param lr:
        :param visualise_batch
        :return:
        """
        training_results = defaultdict(list)
        dataset_manager = self.dataset_loader(
            data_path=self.path_to_training_data,
            augmentation=self.augmentation,
            batch_size=batch
        )
        try:
            image_dataset, data_loaders, dataset_sizes, class_names =\
                                            dataset_manager.generate_training_datasets()
        except Exception as e:
            print(f"Failed to generate training dataset. Error: {e}")
            raise Exception

        if visualise_batch:
            self.visualise_batch(data_loaders["train"])

        # main training loop
        for model_name in models_to_train:
            print(f"\nPreparing model {model_name}. Pretrained: {self.pretrained}")
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

            # If not fine tuning -> freeze
            if not self.fine_tuning:
                model = self.freeze_model_layers(model)
                print(f"{model_name}'s layers frozen. Fine tuning: {self.fine_tuning}")

            # Reshape model depending on the number of classes to learn. New classifier's
            # parameters have .requires_grad = True
            if self.nb_of_classes != 1000:
                model = self.reshape_model(model)
                print(f"{model_name}'s been reshaped to predict {self.nb_of_classes} classes")

            # Ensure model is in the memory of selected device
            try:
                model.to(self.device)
            except Exception as e:
                print(f"Failed during moving model to {self.device}. Error: {e}")
                raise Exception

            # Collect model's parameters to train
            parameters_to_train = self.get_params_to_train(model)

            # Initialize optimizer to perform gradient descend
            if optimizer_name.upper() == "SGD":
                optimizer = optim.SGD(params=parameters_to_train, lr=lr, momentum=0.9)
            elif optimizer_name.upper() == "ADAM":
                optimizer = optim.Adam(params=parameters_to_train, lr=lr, betas=(0.9, 0.999))
            else:
                raise NotImplementedError("Check your optimizer. You can select either Adam or SGD")

            # Initialize loss function
            loss_function = nn.CrossEntropyLoss()

            # Initialize schedule to degrade lr during training
            scheduler = None
            if scheduler_required:
                scheduler = optim.lr_scheduler.StepLR(
                    optimizer=optimizer,
                    step_size=7,
                    gamma=0.1
                )

            # Inception's input size is different, generate new dataset for the model
            is_inception = False
            if model_name == "inception3":
                print("Regenerating new dataset for Inception3")
                dataset_manager = self.dataset_loader(
                    data_path=self.path_to_training_data,
                    augmentation=self.augmentation,
                    input_size=299,
                    batch_size=batch
                )
                image_dataset, data_loaders, dataset_sizes, class_names = \
                                        dataset_manager.generate_training_datasets()
                is_inception = True

            trained_model, performance_metrics = self.train(
                model=model,
                model_name=model_name,
                loss_function=loss_function,
                optimizer=optimizer,
                scheduler=scheduler,
                dataloaders=data_loaders,
                dataset_sizes=dataset_sizes,
                epochs=epochs,
                is_inception=is_inception
            )

            # Save model's training results
            training_results[model_name].append(performance_metrics)

            # Save actual model or state_dict
            self.save_model(
                model=trained_model,
                model_name=model_name,
                accuracy=round(max(performance_metrics[0]), 4),
                optimizer=optimizer_name
            )

            # Better safe than sorry
            del model, trained_model, performance_metrics, parameters_to_train
            torch.cuda.empty_cache()

        print("\nTraining complete. All weights saved to:", self.save_path)
        self.print_out_training_results(training_results)
        self.visualize_training_results(training_results)
