import torch
from torch.nn import Module
from torchvision import transforms
import torchvision
import sys
import cv2
from typing import List, Dict
import torch.nn.functional as F
from PIL import Image
import numpy as np


'''
TODO:
1. Fix your bug
2. Implement method that draws boxes using the provided coordinates. The defected ones color red
3. Test
'''


class TheModelClass(Module):
    pass


class TrainedModel():
    """

    """
    def __init__(
            self,
            load_type,
            path_to_data,
            classes,
            batch_size=10,
            model_class=None
    ):
        # Load entire model
        if load_type == "model":
            print("Attempting to load a model...")
            try:
                self.model = torch.load(path_to_data)
                self.model.eval()
                self.model.cuda()
            except Exception as e:
                print(f"Failed to load the model: {path_to_data}. Error: {e}")
                raise
            print("Model initialized")

        # Load state dict
        else:
            # Use model class and statedict data provided to load the model
            raise NotImplementedError

        self.classes = classes
        self.batch_size = batch_size

    def predict_batch(self, batch_of_images: List[Image.Image]) -> list:
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
        try:
            images_batch = images_batch.cuda()
        except Exception as e:
            print(f"Failed to move batch to GPU. Error: {e}")
            raise

        # forward pass
        with torch.no_grad():
            model_output = self.model(images_batch)

        # parse predictions
        labels = [self.classes[out.data.numpy().argmax()] for out in model_output.cpu()]

        return labels

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
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

    def _coord_preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        image_transforms = torchvision.transforms.Compose([
            transforms.Resize((255, 255)),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )]
        )
        return image_transforms(image)

    def predict_using_coord(
            self,
            images_on_gpu: torch.Tensor,
            test_cases: list
    ) -> None:
        """

        :param images_on_gpu:
        :param test_cases:
        :return:
        """
        # TODO: Batch size

        # IMPORTANT: It is assumed test cases and images on GPU come in the order when test case n
        # belongs to images_on_gpu[n-1], etc.
        assert len(images_on_gpu) == len(test_cases), "Nb of images != test cases"

        dumpers_to_process = list()
        for i, test_case in enumerate(test_cases):
            image = images_on_gpu[i]
            for key, value in test_case.dumpers.items():
                coord = value["coord"]
                assert isinstance(coord, list), "Something went wrong with coordinates"

                # Crop out dumper using its coordinates on the image
                top = coord[0]
                bottom = coord[1]
                left = coord[2]
                right = coord[3]
                subimage = image[:, top:bottom, left:right]

                # Resize an image and append to images to process
                subimage = subimage.unsqueeze(dim=0)  # interpolate requires extra dim
                resized_subimage = F.interpolate(subimage, size=(256, 256))
                dumpers_to_process.append(resized_subimage)

        # Create a batch
        try:
            batch = torch.cat(dumpers_to_process)
        except Exception as e:
            print(f"Failed during batch concatination. Error: {e}")
            raise

        # Visualize slices subimages
        #TrainedModel.visualise_sliced_img(dumpers_to_process)

        # Run NN, get predictions
        predictions = self.run_forward_pass(batch)

        # Loop over all test cases and match the results
        assert len(predictions) == sum(e.nb_of_dumpers for e in test_cases), \
                                    "Nb of predicts doesn't match the number of test dumpers"
        result_index = 0
        for test_case in test_cases:
            nb_of_dumpers = test_case.nb_of_dumpers
            for i in range(nb_of_dumpers):
                test_case.dumpers[i]["defected"] = predictions[result_index]
                result_index += 1


    def run_forward_pass(self, images_on_gpu: torch.Tensor) -> list:
        with torch.no_grad():
            model_output = self.model(images_on_gpu)

        labels = [self.classes[out.data.numpy().argmax()] for out in model_output.cpu()]

        return labels

    @staticmethod
    def visualise_sliced_img(images: List[torch.Tensor]) -> None:
        for image in images:
            image = image.squeeze()
            image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #plt.imshow(image)
            cv2.imshow("window", image_rgb)
            cv2.waitKey(0)
