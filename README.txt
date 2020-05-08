Built-in PyTorch Neural Nets Trainer

Python/Torch module that allows to take neural networks already implemented in the PyTorch framework
and train them to solve your problem. All you need is your own custom dataset in the ImageFolder
format. Gives a good idea as to what nets perform better for your problem, so that you can go ahead
and manually tune and test the promising ones.

Likely to work:
1. Pretrained, fine tuning. Likely will overfit
2. Pretrained, feature extraction. Might be not very accurate, vib
dumpers were not in the ImageNet
3. Quantized transfer learning - freeze the conv layers, train the
head only. Unfreeze the model, drop LR, continue.


To play with:
- Learning rate
- Optimizer
- Batch size
- Fine tuning VS feature extraction VS Quantized transfer learning
- Scheduler
- Augmentation
- Different nets