from torchvision import datasets, transforms
import numpy as np
import torch
from PIL import Image


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
    def validation(model, data_loaders, device):
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
