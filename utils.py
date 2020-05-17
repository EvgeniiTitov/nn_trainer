import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os


class BBDrawer:

    @staticmethod
    def draw_bbs(image: np.ndarray, objs: dict) -> None:
        for key, value in objs.items():
            coordinates = value["coord"]
            defeciency_status = value["defected"]
            top = coordinates[0]
            bot = coordinates[1]
            left = coordinates[2]
            right = coordinates[3]
            colour = (0, 255, 0) if defeciency_status == "ok" else (0, 0, 255)
            cv2.rectangle(image, (left, top), (right, bot), colour, thickness=4)

        # cv2.imshow("", image)
        # cv2.waitKey(0)

    @staticmethod
    def save_image(image: np.ndarray, save_path: str, name: str) -> None:
        save_name = os.path.join(save_path, name)
        try:
            cv2.imwrite(save_name, image)
        except Exception as e:
            print(f"Failed to save image: {name}. Error: {e}")


class Visualizer:

    @staticmethod
    def visualize_models_performance(models_performance: dict):
        """

        :param models_performance:
        :return:
        """
        labels = list()
        for model_name, performance_metrics in models_performance.items():
            # Metrics were returned in a tuple. Then appended to defaultdict(list), so [][]
            accuracy = performance_metrics[0][0]
            loss = performance_metrics[0][1]
            early_stopping = performance_metrics[0][2]

            plt.subplot(1, 2, 1)
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.plot(accuracy, linewidth=3)

            plt.subplot(1, 2, 2)
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.plot(loss, linewidth=3)

            labels.append(model_name)

        plt.legend(labels)
        plt.show()

    @staticmethod
    def visualize_training_results(accuracy, loss):
        """

        :param accuracy:
        :param loss:
        :return:
        """
        if len(accuracy) > 0 and len(loss) > 0:
            plt.subplot(1, 2, 1)
            plt.plot(accuracy, '-go')
            plt.title("Accuracy")

            plt.subplot(1, 2, 2)
            plt.plot(loss, '-ro')
            plt.title("Loss")

            plt.show()

        return

    def model_visualisation(
            self,
            model,
            nb_of_images,
            data_loaders,
            device,
            class_names
    ):
        """

        :param model:
        :param nb_of_images:
        :param data_loaders:
        :param device:
        :param class_names:
        :return:
        """
        was_training = model.training
        model.eval()
        images_processed = 0
        figure = plt.figure()

        with torch.no_grad():
            for i, (batch_of_images, labels) in enumerate(data_loaders["val"]):
                batch = batch_of_images.to(device)
                labels = labels.to(device)

                activations = model(batch)
                # classes_predicted: tensor([0, 1, 1, 1], device='cuda:0')
                _, classes_predicted = torch.max(activations, dim=1)

                # batch.size(): torch.Size([4, 3, 224, 224])
                for i in range(batch.size()[0]):
                    images_processed += 1

                    ax = plt.subplot(nb_of_images // 2, 2, images_processed)
                    ax.axis("off")

                    ax.set_title(f"Predicted: {class_names[classes_predicted[i]]}")

                    self.show(batch.cpu().data[i])

                    if images_processed == nb_of_images:
                        model.train_models(mode=was_training)
                        return

        model.train_models(mode=was_training)

    @staticmethod
    def print_out_training_results(training_results):
        """

        :param training_results:
        :return:
        """
        print("\nTRAINING RESULTS:")
        for model, performance_result in training_results.items():
            print(
                'Model:{} Best acc: {:.4f} on {} epoch'.format(
                    model,
                    performance_result[0][2],
                    performance_result[0][3]
                )
            )

        return

    @staticmethod
    def show(image, title=None):
        """

        :param image:
        :param title:
        :return:
        """
        img = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        if title is not None:
            plt.title(title)
        plt.show()
        plt.pause(0.001)
