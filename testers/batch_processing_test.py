from model.model import TrainedModel
from random import shuffle
import os
import sys
from PIL import Image
import time
import inspect


BATCH_SIZE = 10


'''
1. Find bug (batch size affects accuracy + seems like it keeps running over one batch
2. Test
3. Add ability to save images with correct predictions 
'''


class BatchTester:
    counter = 0

    @staticmethod
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

    @staticmethod
    def collect_test_images(folder) -> list:
        return [os.path.join(folder, image) for image in os.listdir(folder)]

    @staticmethod
    def determine_if_defected(image_name) -> str:
        if "def" in image_name:
            return "def"
        elif "ok" in image_name:
            return "ok"
        else:
            return ""

    @staticmethod
    def prep_images_labels(path_to_images: list) -> tuple:
        labels, decoded_images = [], []
        for path_to_image in path_to_images:
            # Read and save image label
            image_name = os.path.basename(path_to_image)
            defect_status = BatchTester.determine_if_defected(image_name)
            if not defect_status:
                print(f"Check your image name: {image_name}")
                continue
            labels.append(defect_status)
            # Open image itself
            try:
                image = Image.open(path_to_image)
                # image = cv2.imread(path_to_image)
            except Exception as e:
                print("Failed to open:", path_to_image, "Error:", e)
                labels.pop()  # pop label associated with the failed image
                continue
            decoded_images.append(image)

        assert len(labels) == len(decoded_images), "Nb of images != labels"

        return labels, decoded_images

    @staticmethod
    def calculate_correct_pred(predictions, labels):
        correct = 0
        assert len(predictions) == len(labels), "Nb of predictions != labels"
        for i in range(len(predictions)):
            if predictions[i] == labels[i]:
                correct += 1

        return correct

    @staticmethod
    def save_processed_batch(batch, labels, save_path):
        assert len(batch) == len(labels), "Nb of images != labels"

        for i in range(len(batch)):
            image = batch[i]
            label = labels[i]
            image_name = str(BatchTester.counter) + '_' + label
            BatchTester.counter += 1
            try:
                image.save(os.path.join(save_path, image_name + ".jpg"))
            except Exception as e:
                print(f"Failed while saving processed image. Error: {e}")
                continue


def generate_test_images(path_to_images, class_, destination):
    assert os.path.isdir(path_to_images), "Provided folder is not a folder"

    for image in os.listdir(path_to_images):
        if not any(image.endswith(ext.lower()) for ext in ["jpg", "png", "jpeg"]):
            continue
        new_name = class_ + "_" + image
        os.rename(
            os.path.join(path_to_images, image),
            os.path.join(destination, new_name)
        )


def main():
    #print(inspect.getsource(Image))

    #path_to_model = r"D:\Desktop\system_output\dumper_training\resnet18_Acc1.0_Ftuned1_Pretrained1_OptimizerADAM.pth"
    path_to_model = r"D:\Desktop\system_output\dumper_training\resnet18_Acc0.8249_Ftuned1_Pretrained1_OptimizerSGD.pth"
    folder_images = r"D:\Desktop\Programming\ML_NN\DataSets\test_dumpers"
    save_path = r"D:\Desktop\system_output\TEST_DUMPERS"
    classes = ["def", "ok"]

    # Load a model
    model = TrainedModel(
        load_type="model",
        path_to_data=path_to_model,
        classes=classes
    )

    # Load images, form into batches
    test_images_paths = BatchTester.collect_test_images(folder_images)
    shuffle(test_images_paths)
    labels, images = BatchTester.prep_images_labels(test_images_paths)
    nb_test_images = len(images)
    print("Nb of test images:", nb_test_images)

    # Create batches of images with labels
    image_batches = BatchTester.split_into_batches(images, batch_size=BATCH_SIZE)
    label_batches = BatchTester.split_into_batches(labels, batch_size=BATCH_SIZE)

    assert len(image_batches) == len(label_batches), "Nb of image batches != label bathes"
    print("Nb of batches:", len(image_batches))

    correct_pred = 0
    # Run nn for each batch, get predictions, calculate error
    t_start = time.time()
    for i in range(len(image_batches)):
        batch_of_images = image_batches[i]
        batch_of_labels = label_batches[i]

        # Send batch of images to NN
        predictions = model.predict_batch(batch_of_images)

        # Save processed images
        BatchTester.save_processed_batch(batch_of_images, predictions, save_path)

        # Compare true labels with the predicted ones
        corrects = BatchTester.calculate_correct_pred(predictions, batch_of_labels)
        correct_pred += corrects
        print("Correct predictions:", corrects)
        print()

    time_taken = time.time() - t_start
    print(f"\nTIME TAKEN WITH BATCH SIZE {BATCH_SIZE} is {round(time_taken, 3)} seconds")

    # Calculate error
    accuracy = correct_pred / nb_test_images
    print(f"\nACCURACY ON {nb_test_images} IMAGES: {round(accuracy, 3)}")

if __name__ == "__main__":
    main()
