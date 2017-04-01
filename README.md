# deep-learning-films-genre
As part of a deep learning course project, we decide to use neural network to predict film genre from its poster


## Procedure

### Data Loading
First if you don't get the posters, run : `dataset_preparation.py`
It will load all images but will take a long time (depending on your internet connection). Then it will create a file "labels.json" which
represents a dictionary where keys are name of the films and values are a list of genres of the films.

### Creation of training and test set
To create three sets, a training one, a validation one and a test one, run : `train_val_test_set_creation.py`

### Test if data can be load
You can run `python -m unittest test_data_loading` to test if all data can be load by Dataset_Manager (in file "dataset_manager.py")

### Finetune AlexNet
By running `finetune.py`, the training of the last layer of AlexNet and the finetuning of all its layers are beginning. You now just have to wait, depending on your configuration

### Logs of your training
When your traning finishes, go to the "logs" folder to see your logs (training loss, mean average precision)

### Save model
Your model will be saved in saved_models by using early stopping. 

