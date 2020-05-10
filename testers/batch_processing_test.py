from model.model import TrainedModel
import os
import cv2


def calculate_accuracy():
    pass


def create_batches(images, batch_size=6):
    assert len(images) >= batch_size, "Batch size cannot be greater than nb of images"
    if len(images) < batch_size:
        return images

    batches = list()
    batch = list()
    while images:
        image = images.pop()
        batch.append(image)

        if len(batch) == batch_size:
            batches.append(batch)
            batch = list()

    # In case not divisible
    if batch:
        batches.append(batch)

    return batches


def collect_test_case(folder):
    test_images = list()
    for element in os.listdir(folder):
        test_images.append(os.path.join(folder, element))

    return test_images


def open_images(images):
    decoded_images = list()
    for path_to_image in images:
        try:
            image = cv2.imread(path_to_image)
            decoded_images.append(image)
        except Exception as e:
            print("Failed to open:", path_to_image, "Error:", e)
            continue

    return decoded_images


def main():
    path_to_model = r""
    path_to_images = r"D:\Desktop\system_output\TEST_DUMPERS"
    labels = ["defected", "fine"]

    test_images_paths = collect_test_case(path_to_images)
    images = open_images(test_images_paths)
    batches = create_batches(images, batch_size=6)

    # How do we track predictions? We need to collect labels as well
    # Train model, load here, test it
    # Probably write a quick script to rename all your valid images to simulate
    # prod situation and test accuracy properly

    #model_instance = TrainedModel("model", path_to_model, labels)


if __name__ == "__main__":
    main()
