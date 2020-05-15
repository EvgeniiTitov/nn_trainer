from model.model import TrainedModel
from torchvision import transforms
import torchvision
import torch
from testers.batch_processing_test import BatchTester
from PIL import Image
import sys
import numpy as np
import cv2


'''
Once images on GPU, send coordinates and references to images on GPU to your model,
so that it can make predictions on images already loaded into GPU. There's no need to
send actual images to your model if they've already been uploaded to your device
'''


class HostDeviceOptimizer:
    """

    """
    @staticmethod
    def load_images_to_GPU(images: list) -> torch.Tensor:
        image_tensors = list()
        for image in images:
            image_tensor = HostDeviceOptimizer._preprocess_image(image)
            image_tensor.unsqueeze_(0)
            image_tensors.append(image_tensor)

        batch = torch.cat(image_tensors)
        try:
            batch_gpu = batch.cuda()
            print("Images successfully moved from host to device")
            return batch_gpu
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
    def read_images(paths: list) -> list:
        images = []
        for path_to_image in paths:
            try:
                image = Image.open(path_to_image)
            except Exception as e:
                print(f"Failed to open image {path_to_image}. Error: {e}")
                continue
            images.append(image)

        return images


def main():
    path_to_model = r"D:\Desktop\system_output\dumper_training\decent\resnet18_Acc1.0_Ftuned1_Pretrained1_OptimizerADAM.pth"
    folder_images = r"D:\Desktop\FutureLab\test_dumpers"
    save_path = r"D:\Desktop\system_output\TEST_DUMPERS"
    classes = ["def", "ok"]

    # Load a model
    model = TrainedModel(
        load_type="model",
        path_to_data=path_to_model,
        classes=classes
    )

    dumpers_coordinates = {
        "DJI_0274_1300.jpg":
            {
                1: [184, 282, 1163, 1355],
                2: [246, 341, 1679, 1806]
            },
        "DJI_0274_2500.jpg":
            {
                1: [580, 726, 1453, 1718],
                2: [568, 697, 480, 750]
            },
        "DJI_0278_3000.jpg":
            {
                1: [544, 682, 1353, 1594],
                2: [601, 750, 495, 792]
            }
    }

    # coordinates2 = {
    #     "IMG_1743.JPG":
    #         {
    #             1: [2347, 2697, 296, 407],
    #             2: [904, 1262, 1980, 2082],
    #             3: [2330, 2686, 1916, 2018],
    #             4: [1051, 1387, 369, 474],
    #             5: [1073, 1496, 3139, 3236],
    #             6: [2551, 2939, 3095, 3199],
    #             7: [2260, 2669, 714, 817]
    #         }
    # }

    test_images_paths = BatchTester.collect_test_images(folder_images)
    images = HostDeviceOptimizer.read_images(test_images_paths)
    images_gpu = HostDeviceOptimizer.load_images_to_GPU(images)
    result = model.predict_using_coord(images_gpu, dumpers_coordinates)


if __name__ == "__main__":
    main()
