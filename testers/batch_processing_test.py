from model.model import TrainedModel
from random import shuffle
import os
import cv2
import sys
from PIL import Image

BATCH_SIZE = 6


def calculate_accuracy():
    pass


def split_into_batches(items, batch_size=6):
    """
    Splits images/labels into batches
    :param items:
    :param batch_size:
    :return:
    """
    assert len(items) >= batch_size, "Batch size cannot be greater than nb of images/labels"

    if len(items) < batch_size:
        return items

    batches = list()
    batch = list()
    while items:
        item = items.pop()
        batch.append(item)

        if len(batch) == batch_size:
            batches.append(batch)
            batch = list()

    # In case not divisible
    if batch:
        batches.append(batch)

    return batches


def collect_test_images(folder) -> list:
    return [os.path.join(folder, image) for image in os.listdir(folder)]


def determine_if_defected(image_name) -> str:
    if "def" in image_name:
        return "def"
    elif "ok" in image_name:
        return "ok"
    else:
        return ""


def prep_images_labels(path_to_images: list) -> tuple:
    decoded_images = list()
    labels = list()
    for path_to_image in path_to_images:
        # Read and save image label
        image_name = os.path.basename(path_to_image)
        defect_status = determine_if_defected(image_name)
        if not defect_status:
            print(f"Check your image name: {image_name}")
            continue
        labels.append(defect_status)

        # Open image itself
        try:
            image = Image.open(path_to_image)
            #image = cv2.imread(path_to_image)
        except Exception as e:
            print("Failed to open:", path_to_image, "Error:", e)
            labels.pop()  # pop label associated with the failed image
            continue
        decoded_images.append(image)

    assert len(labels) == len(decoded_images), "Nb of images and labels !="

    return labels, decoded_images


def main():
    accuracy = 0.0
    path_to_model = r""
    path_to_images = r"D:\Desktop\system_output\TEST_DUMPERS"
    classes = ["defected", "fine"]

    # Load a model
    model = TrainedModel(
        load_type="model",
        path_to_data=path_to_model,
        classes=classes
    )

    # Load images, form into batches
    test_images_paths = collect_test_images(path_to_images)
    shuffle(test_images_paths)
    labels, images = prep_images_labels(test_images_paths)

    # Creates batches [[(label, image),...], ...]
    image_batches = split_into_batches(images, batch_size=BATCH_SIZE)
    label_batches = split_into_batches(labels, batch_size=BATCH_SIZE)
    assert len(image_batches) == len(label_batches), "Nb of image batches != label bathes"

    for i in range(len(image_batches)):
        batch_of_images = image_batches[0]
        batch_of_labels = label_batches[0]

        # Send batch of images to NN
        predictions = model.predict_batch(batch_of_images)

        # Compare true labels with the predicted ones



if __name__ == "__main__":
    main()
