import torch
from torch.nn import Module
from torchvision import transforms
import torchvision
import sys


class TheModelClass(Module):
    pass


class TrainedModel():
    """

    """
    def __init__(self, load_type, path_to_data, classes, model_class=None):
        # Load entire model
        if load_type == "model":
            print("Attempting to load a model...")
            try:
                self.model = torch.load(path_to_data)
                self.model.eval()
                self.model.cuda()
            except Exception as e:
                print("Failed to load the model:", path_to_data)
                raise
            print("Model initialized")

        # Load state dict
        else:
            # Use model class and statedict data provided to load the model
            raise NotImplementedError

        self.classes = classes

    def predict_batch(self, batch_of_images: list):
        """

        :param images:
        :return:
        """
        image_tensors = list()

        # preprocess images and construct a batch
        for image in batch_of_images:
            image_tensor = self._preprocess_image(image)
            image_tensor.unsqueeze_(0)
            image_tensors.append(image_tensor)

        images_batch = torch.cat(image_tensors)
        images_batch = images_batch.cuda()

        # forward pass
        with torch.no_grad():
            model_output = self.model(images_batch)

        # parse predictions
        labels = [self.classes[out.data.numpy().argmax()] for out in model_output.cpu()]

        return labels

    def _preprocess_image(self, image):
        """

        :param image:
        :return:
        """
        image_transforms = torchvision.transforms.Compose([
                    transforms.Resize((255, 255)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    )]
        )
        return image_transforms(image)

    def predict_using_coord(self, images_on_gpu, coordinates):

        # 1. Define regions on the images using coordinates that need to be run through
        # the network.

        # 2. Combine them in a batch



        with torch.no_grad():
            model_output = self.model(images_on_gpu)

        labels = [self.classes[out.data.numpy().argmax()] for out in model_output.cpu()]

        return labels