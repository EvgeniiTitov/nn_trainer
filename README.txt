commands:

TRAINING:
--train= path to the folder with training images (ImageFolder format) 
--learning_type= 0 - only new classifier gets trained, 1 - all layers will be trained
--save_weights= path to where save parameters

TESTING:
--test= path to an image or a folder with images to get processed
--weights= path to parameters(weights) to use during testing

EXTRA TRAINING FUNCTIONALITY:
--gpu= True (by default it is True, gets changed to cpu in case gpu is not available)
--epoch= number of epoch
--draw_metircs= 1-True/False (draw accuracy and loss on val dataset for each training epoch)
--visualize= 1-True/False (make actual predictions on some images from the val dataset, show them)



training example:
python C:\Users\Evgenii\Desktop\Machine_Learning_NNs\work_related_models\renet18.py 
--train=D:\Desktop\Reserve_NNs\IMAGES_ROW_DS\DEFECTS\craks_dataset --learning_type=1 
--epoch=30 --draw_metrics=1

