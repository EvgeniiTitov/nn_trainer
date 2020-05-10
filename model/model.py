import torch
from torch.nn import Module
from torchvision import transforms
import torchvision

class TheModelClass(Module):
    pass


class Model():
    """

    """
    def __init__(self, save_type, path_to_data, labels, model_class=None):
        # Load entire model
        if save_type == "model":
            try:
                self.model = torch.load(path_to_data)
                self.model.eval()
            except Exception as e:
                print("Failed to load the model:", path_to_data)
                raise
            print("Model initialized")

        # Load state dict
        else:
            # Use model class and statedict data provided to load the model
            raise NotImplementedError

        self.labels = labels

    def predict_batch(self, images):
        """

        :param images:
        :return:
        """
        image_tensors = list()

        # preprocess images and construct a batch
        for image in images:
            image_tensor = self.preprocess_image(image)
            image_tensor.unsqueeze_(0)
            image_tensors.append(image_tensor)

        images_batch = torch.cat(image_tensors)

        # forward pass
        with torch.no_grad():
            model_output = self.model(images_batch)

        # parse predictions
        labels = [self.labels[out.data.numpy().argmax()] for out in model_output]

        return labels

    def preprocess_image(self, image):
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



