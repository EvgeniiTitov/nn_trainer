from model.model import TrainedModel
from torchvision import transforms
from typing import List
import torchvision
import torch
from utils import BBDrawer
from testers.batch_processing_test import BatchTester
from PIL import Image
import sys
import numpy as np
import cv2
import os


'''
Once images on GPU, send coordinates and references to images on GPU to your model,
so that it can make predictions on images already loaded into GPU. There's no need to
send actual images to your model if they've already been uploaded to your device.

Your images stay on GPU, you send references to those images and coordinates between
your workers instead of actual images.
'''


class TestImage:
    def __init__(self, image_name: str, dumpers: List[list]):
        self.name = image_name
        self.nb_of_dumpers = len(dumpers)

        self.dumpers = dict()
        for i, dumper in enumerate(dumpers):
            self.dumpers[i] = {
                "coord": dumper,
                "defected": None
            }

    def __str__(self):
        return f"Name: {self.name}, Nb of dumpers: {self.nb_of_dumpers}"


class HostDeviceOptimizer:
    """

    """
    @staticmethod
    def load_images_to_GPU(images: List[tuple]) -> tuple:
        """
        :param images:
        :return:
        """
        image_tensors = list()
        image_names = list()
        for image_name, image in images:
            image_tensor = HostDeviceOptimizer._preprocess_image(image)
            image_tensor.unsqueeze_(0)
            image_tensors.append(image_tensor)
            image_names.append(image_name)

        batch = torch.cat(image_tensors)
        try:
            batch_gpu = batch.cuda()
            print("Images successfully moved from host to device")
            return (image_names, batch_gpu)
        except Exception as e:
            print(f"Moving images to GPU failed. Error: {e}")
            raise

    @staticmethod
    def _preprocess_image(image: Image.Image) -> torch.Tensor:
        """

        :param image:
        :return:
        """
        image_transforms = torchvision.transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]
                    )]
        )
        return image_transforms(image)

    @staticmethod
    def read_images(paths: List[str]) -> List[tuple]:
        """
        Opens images using PIL.Image
        :param paths:
        :return:
        """
        images = []
        for path_to_image in paths:
            image_name = os.path.split(path_to_image)[-1]
            try:
                #image = Image.open(path_to_image)
                image = cv2.imread(path_to_image)
            except Exception as e:
                print(f"Failed to open image {path_to_image}. Error: {e}")
                continue
            images.append((image_name, image))

        return images


def main():
    path_to_model = r"D:\Desktop\system_output\dumper_training\decent\resnet18_Acc1.0_Ftuned1_Pretrained1_OptimizerADAM.pth"
    folder_images = r"D:\Desktop\FutureLab\test_dumpers"
    save_path = r"D:\Desktop\FutureLab\processed_dumpers"
    classes = ["def", "ok"]

    # Load a model
    model = TrainedModel(
        load_type="model",
        path_to_data=path_to_model,
        classes=classes
    )

    test_cases = [
        ("12_1175.jpg", [[1008, 1056, 1122, 1222], [1006, 1056, 762, 839]]),
        ("12_5200.jpg", [[355, 409, 859, 963], [350, 402, 483, 565]]),
        ("13_5450.jpg",[[377, 427, 404, 479]]),
        ("DJI_0108_1200.jpg", [[227, 276, 935, 1033], [178, 232, 1341, 1426]]),
        ("DJI_0108_3100.jpg", [[85, 120, 1703, 1774], [78, 114, 1442, 1523], [648, 691, 1279, 1359]]),
        ("DJI_0109_2450.jpg", [[823, 913, 1504, 1656], [914, 1001, 924, 1098]]),
        ("DJI_0112_3600.jpg", [[247, 314, 1240, 1371]]),
        ("DJI_0198_1450.jpg", [[906, 999, 1044, 1193], [889, 976, 567, 707], [113, 167, 1592, 1669],
                               [688, 738, 1658, 1752], [158, 210, 1367, 1431]]),
        ("DJI_0214_700.jpg", [[495, 592, 806, 1004], [545, 650, 1474, 1635]]),
        ("DJI_0217_600.jpg", [[484, 620, 76, 287], [559, 668, 924, 1114]]),
        ("DJI_0224_1150.jpg", [[564, 659, 119, 332], [648, 760, 779, 960]]),
        ("DJI_0249_2300.jpg", [[191, 288, 1339, 1528], [228, 291, 813, 919]]),
        ("DJI_0229_2600.jpg", [[829, 923, 823, 1035], [878, 977, 1467, 1623]])

    ]

    test_instances = list()
    for name, dumpers in test_cases:
        test_instances.append(TestImage(name, dumpers))

    # get paths to test images
    test_images_paths = BatchTester.collect_test_images(folder_images)
    # get a list of matrices (opened images)
    names_images = HostDeviceOptimizer.read_images(test_images_paths)
    # convert imgs into torch.Tensor, .cat() them and move to GPU
    image_names, images_gpu = HostDeviceOptimizer.load_images_to_GPU(names_images)

    model.predict_using_coord(images_gpu, test_instances)

    for index, test_case in enumerate(test_instances):
        image_name, image = names_images[index]
        dumpers = test_case.dumpers
        BBDrawer.draw_bbs(image, dumpers)
        BBDrawer.save_image(image, save_path, image_name)


if __name__ == "__main__":
    main()

    dumpers_coordinates = {
        "12_1175.jpg":
            {
                1: [1008, 1056, 1122, 1222],
                2: [1006, 1056, 762, 839]
            },
        "12_5200.jpg":
            {
                1: [355, 409, 859, 963],
                2: [350, 402, 483, 565]
            },
        "13_5450.jpg":
            {
                1: [377, 427, 404, 479]
            },
        "DJI_0108_1200.jpg":
            {
                1: [227, 276, 935, 1033],
                2: [178, 232, 1341, 1426]
            },
        "DJI_0108_3100.jpg":
            {
                1: [85, 120, 1703, 1774],
                2: [78, 114, 1442, 1523],
                3: [648, 691, 1279, 1359]
            },
        "DJI_0109_2450.jpg":
            {
                1: [823, 913, 1504, 1656],
                2: [914, 1001, 924, 1098]
            },
        "DJI_0112_3600.jpg":
            {
                1: [247, 314, 1240, 1371]
            },
        "DJI_0198_1450.jpg":
            {
                1: [906, 999, 1044, 1193],
                2: [889, 976, 567, 707],
                3: [113, 167, 1592, 1669],
                4: [688, 738, 1658, 1752],
                5: [158, 210, 1367, 1431]
            },
        "DJI_0214_700.jpg":
            {
                1: [495, 592, 806, 1004],
                2: [545, 650, 1474, 1635]
            },
        "DJI_0217_600.jpg":
            {
                1: [484, 620, 76, 287],
                2: [559, 668, 924, 1114]
            },
        "DJI_0224_1150.jpg":
            {
                1: [564, 659, 119, 332],
                2: [648, 760, 779, 960]
            },
        "DJI_0229_2600.jpg":
            {
                1: [829, 923, 823, 1035],
                2: [878, 977, 1467, 1623]
            },
        "DJI_0249_2300.jpg":
            {
                1: [191, 288, 1339, 1528],
                2: [228, 291, 813, 919]
            }
    }