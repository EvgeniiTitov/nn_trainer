from model.model import TrainedModel
from torchvision import transforms
from typing import List
import torchvision
import torch
from utils import BBDrawer
from testers.batch_processing_test import BatchTester
from PIL import Image
import cv2
import os
import sys


'''
Once images on GPU, send coordinates and references to images on GPU to your model,
so that it can make predictions on images already loaded into GPU. There's no need to
send actual images to your model if they've already been uploaded to your device.

Your images stay on GPU, you send references to those images and coordinates between
your workers instead of actual images.
'''


class TestImage:
    """
    Simulates the object representation format used in the app
    """
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
            return image_names, batch_gpu
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
        ("13_5450.jpg", [[377, 427, 404, 479]]),
        ("DJI_0108_1200.jpg", [[227, 276, 935, 1033], [178, 232, 1341, 1426]]),
        ("DJI_0108_3100.jpg", [[85, 120, 1703, 1774], [78, 114, 1442, 1523], [648, 691, 1279, 1359]]),
        ("DJI_0109_2450.jpg", [[823, 913, 1504, 1656], [914, 1001, 924, 1098]]),
        ("DJI_0112_3600.jpg", [[247, 314, 1240, 1371]]),
        ("DJI_0198_1450.jpg", [[906, 999, 1044, 1193], [889, 976, 567, 707], [113, 167, 1592, 1669],
                               [688, 738, 1658, 1752], [158, 210, 1367, 1431]]),
        ("DJI_0214_700.jpg", [[495, 592, 806, 1004], [545, 650, 1474, 1635]]),
        ("DJI_0217_600.jpg", [[484, 620, 76, 287], [559, 668, 924, 1114]]),
        ("DJI_0224_1150.jpg", [[564, 659, 119, 332], [648, 760, 779, 960]]),
        ("DJI_0229_2600.jpg", [[829, 923, 823, 1035], [878, 977, 1467, 1623]]),
        ("DJI_0249_2300.jpg", [[191, 288, 1339, 1528], [228, 291, 813, 919]]),
        ("DJI_0250_1600.jpg", [[707, 828, 227, 476], [741, 859, 1020, 1233]]),
        ("DJI_0252_5900.jpg", [[727, 833, 1103, 1315], [777, 871, 499, 689]]),
        ("DJI_0252_6050.jpg", [[499, 617, 310, 570], [586, 711, 993, 1177]]),
        ("DJI_0253_1500.jpg", [[818, 981, 1022, 1302], [761, 861, 338, 519]]),
        ("DJI_0259_4150.jpg", [[818, 935, 413, 650], [809, 903, 1229, 1416]]),
        ("DJI_0269_1550.jpg", [[120, 234, 1060, 1314], [161, 251, 384, 553]]),
        ("DJI_0270_1900.jpg", [[638, 834, 430, 747], [544, 666, 1166, 1303]]),
        ("DJI_0272_2250.jpg", [[821, 925, 1503, 1695], [897, 1045, 438, 734]]),
        ("DJI_0273_4100.jpg", [[930, 1047, 1190, 1486], [906, 997, 345, 579]]),
        ("DJI_0275_950.jpg", [[568, 614, 1514, 1608], [651, 702, 713, 793]]),
        ("DJI_0277_2300.jpg", [[805, 903, 312, 526], [760, 835, 922, 1071]]),
        ("DJI_0278_1650.jpg", [[19, 86, 929, 1148]]),
        ("DJI_0284_3650.jpg", [[599, 716, 607, 844], [658, 764, 1326, 1526]]),
        ("DJI_0289_2750.jpg", [[568, 763, 1533, 1920], [535, 718, 467, 795]]),
        ("DJI_0292_3800.jpg", [[693, 817, 167, 429], [694, 812, 1211, 1407]]),
        ("DJI_0322_2600.jpg", [[497, 606, 401, 585], [235, 293, 1680, 1785], [658, 783, 1297, 1508]]),
        ("DJI_0323_1350.jpg", [[366, 466, 266, 461], [343, 455, 1045, 1210]]),
        ("DJI_0326_1150.jpg", [[111, 198, 1572, 1766], [154, 233, 976, 1103], [900, 954, 1823, 1909], [847, 892, 1393, 1454]]),
        ("DJI_0565_2150.jpg", [[424, 512, 706, 880], [520, 603, 1310, 1455]]),
        ("DJI_0568_1700.jpg", [[229, 328, 1072, 1273], [240, 311, 552, 673]]),
        ("DJI_0574_1300.jpg", [[42, 172, 1389, 1681], [46, 164, 412, 643]]),
        ("DJI_0575_1900.jpg", [[659, 774, 303, 575], [695, 809, 1253, 1443]]),
        ("DJI_0576_1850.jpg", [[717, 791, 389, 558], [772, 881, 1012, 1149], [263, 320, 1030, 1114]]),
        ("DJI_0579_1700.jpg", [[232, 282, 764, 858], [698, 763, 756, 871], [716, 778, 249, 388]])
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
    assert image_names == list(e.name for e in test_instances), "Your shitty method doesn't work"
    # run NN for the test cases
    model.predict_using_coord(images_gpu, test_instances)

    for index, test_case in enumerate(test_instances):
        image_name, image = names_images[index]
        dumpers = test_case.dumpers
        BBDrawer.draw_bbs(image, dumpers)
        BBDrawer.save_image(image, save_path, image_name)


if __name__ == "__main__":
    main()
