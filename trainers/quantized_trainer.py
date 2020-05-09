import torchvision.models.quantization as models
from torch import nn
import torch.optim
from trainers.base_trainer import BaseTrainer

'''
Quantized model can only run on the CPU

1. First use a frozen, quantized feature extractor in order to train a new
classifier. The conv layers get frozen so that nothing breaks while the new classier
is training + we do not need to drop the LR as we'd need to if we fine tuned the whole
model.
2. 
'''

class QuantizedTrainer(BaseTrainer):
    def train_model(self, model):
        """

        :param model_to_train:
        :return:
        """
        #
        model_ = self.create_quantized_model(model)
        model_ = model_.to("cpu")
        criterion = nn.CrossEntropyLoss()
        # Only training the classifier
        optimizer = torch.optim.SGD(
            params=model_.parameters(),
            lr=0.01,
            momentum=0.9
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=7,
            gamma=0.1
        )
        trained_model = self.train(
            model=model_,
            model_name="resnet",
            loss_function=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            data_loaders="TBC",
            dataset_sizes="TBC",
            device="TBC",
            epochs="TBC",
            patience="TBC",
            early_stopping="TBC"
        )


    def create_quantized_model(self, model):
        """

        :param model:
        :return:
        """
        num_features = model.fc.in_features
        # 1. Isolate the feature extractor
        model_features = nn.Sequential(
            model.quant,
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
            model.dequant
        )
        # 2. New classifier
        head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_features, self.nb_of_classes)
        )
        # 3. Combine new classifier and feature extractor
        new_model = nn.Sequential(
            model_features,
            nn.Flatten(1),
            head
        )
        return new_model

if __name__ == "__main__":
    model = models.resnet18(
        pretrained=True,
        progress=True,
        quantize=True
    )
    '''
    The pretrained model needs to be modified - it's got the quantize/dequantized blocks
    in the beginning and the end. First we're using only the feature extractor -> the 
    dequantization layer has to move right before the new classifier.
    
    WHEN SEPARATING THE FEATURE EXTRACTOR FROM THE REST OF A QUANTIZED MODEL - MANUALLY PLACE
    THE QUANTIZED/DEQUANTIZED IN THE BEGINNING AND THE END OF THE PARTS YOU WANT TO KEEP 
    QUANTIZED
    '''

