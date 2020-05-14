import torch.optim as optim
from trainers.base_trainer import BaseTrainer
from collections import defaultdict
import torch.nn as nn
import torch

class IndividualTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_model(
            self,
            model_to_train,
            epochs,
            batch_sizes,
            optimizer_names,
            scheduler_required,
            lrs,
            visualise_batch
    ):
        """
        To check:
            batch +- 8
            lr +/- 20%
            optimizer
            scheduler
        :param models_to_train:
        :param epochs:
        :param batch_sizes:
        :param optimizer_names:
        :param scheduler_required:
        :param lrs:
        :param visualise_batch:
        :return:
        """
        training_results = defaultdict(list)
        print("\nTraining commences")
        for optmzr in optimizer_names:
            for batch_size in batch_sizes:
                # Initialize dataloader and generate data for training
                input_size = 224
                is_inception = False
                if model_to_train.lower().strip() == "inception3":
                    print("--> Training the Inception3 model!")
                    is_inception = True
                    input_size = 299

                dataset_manager = self.dataset_loader(
                    data_path=self.path_to_training_data,
                    augmentation=self.augmentation,
                    input_size=input_size,
                    batch_size=batch_size
                )
                try:
                    image_dataset, data_loaders, dataset_sizes, class_names = \
                                                dataset_manager.generate_training_datasets()
                except Exception as e:
                    print(f"Failed to generate training dataset. Error: {e}")
                    raise Exception

                for lr in lrs:
                    print("Optimizer:", optmzr)
                    print("Batch size:", batch_size)
                    print("Learning rate:", lr)

                    model = self.initialize_a_model(model_name=model_to_train)

                    # If not fine tuning -> freeze
                    if not self.fine_tuning:
                        model = self.freeze_model_layers(model)
                        print(f"{model_to_train}'s layers frozen. Fine tuning: {self.fine_tuning}")

                    # Reshape model depending on the number of classes to learn. New classifier's
                    # parameters have .requires_grad = True
                    if self.nb_of_classes != 1000:
                        model = self.reshape_model(model)
                        print(f"{model_to_train}'s been reshaped to predict {self.nb_of_classes} classes")

                    # Ensure model is in the memory of selected device
                    try:
                        model.to(self.device)
                    except Exception as e:
                        print(f"Failed during moving model to {self.device}. Error: {e}")
                        raise Exception

                    # Collect model's parameters to train
                    parameters_to_train = self.get_params_to_train(model)

                    if optmzr.upper().strip() == "SGD":
                        optimizer = optim.SGD(params=parameters_to_train, lr=lr, momentum=0.9)
                    elif optmzr.upper().strip() == "ADAM":
                        optimizer = optim.Adam(params=parameters_to_train, lr=lr, betas=(0.9, 0.999))
                    else:
                        raise NotImplementedError("Check your optimizer. You can select either Adam or SGD")

                    # Initialize loss function
                    loss_function = nn.CrossEntropyLoss()

                    # Initialize the scheduler
                    scheduler = None
                    if scheduler_required:
                        scheduler = optim.lr_scheduler.StepLR(
                            optimizer=optimizer,
                            step_size=7,
                            gamma=0.1
                        )

                    trained_model, performance_metrics = self.train(
                        model=model,
                        model_name=model_to_train,
                        loss_function=loss_function,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        dataloaders=data_loaders,
                        dataset_sizes=dataset_sizes,
                        epochs=epochs,
                        is_inception=is_inception
                    )

                    training_conditions = f"Model: {model_to_train}, " \
                                          f"Optim: {optmzr}, " \
                                          f"Batch size: {batch_size}, lr: {lr}"
                    training_results[training_conditions].append(performance_metrics)

                    # Save actual model or state_dict
                    self.save_model(
                        model=trained_model,
                        model_name=model_to_train,
                        accuracy=round(max(performance_metrics[0]), 4),
                        optimizer=optmzr
                    )

                    # Better safe than sorry
                    del model, trained_model, performance_metrics, parameters_to_train
                    torch.cuda.empty_cache()

        print("\nTraining complete. All weights saved to:", self.save_path)
        self.print_out_training_results(training_results)
        self.visualize_training_results(training_results)
