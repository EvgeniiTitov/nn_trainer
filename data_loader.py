from torchvision import datasets, transforms
import torch
import os


class DatasetLoader:
    """

    """
    def __init__(
            self,
            data_path,
            augmentation,
            input_size=224,
            batch_size=8
    ):

        self.path_to_images = data_path
        self.perform_aug = augmentation
        self.input_size = input_size
        self.batch_size = batch_size

    def generate_transformations(self):
        """
        Data augmentation and normalization for training, only normalization
        for validation
        :return:
        """
        if self.perform_aug:
            print("\nAugmentation will be applied to the training images")
            data_transforms = {
                "train": transforms.Compose([
                    transforms.Resize((self.input_size, self.input_size)),
                    transforms.RandomRotation(degrees=45),
                    transforms.ColorJitter(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                "val": transforms.Compose([
                    transforms.Resize((self.input_size, self.input_size)),  # 256 used to be
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                }
        else:
            print("\nNo augmentation will be applied to the training images")
            data_transforms = {
                "train": transforms.Compose([
                    transforms.Resize((self.input_size, self.input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                "val": transforms.Compose([
                    transforms.Resize((self.input_size, self.input_size)),  # 256 used to be
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            }

        return data_transforms

    def generate_training_datasets(self):
        data_transforms = self.generate_transformations()

        image_datasets = {
            phase: datasets.ImageFolder(os.path.join(self.path_to_images, phase),
                                                     data_transforms[phase])
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
