# deep-learning-films-genre
As part of a deep learning course project, we decide to use neural network to predict film genre from its poster


## Procedure

### Data Loading
First if you don't get the posters, run : `dataset_preparation.py`
It will load all images but will take a long time (depending on your internet connection)

### Creation of training and test set
To create two sets, a training one and a test one, run : `train_test_set_creation.py`

### Test if data can be load
You can run `python -m unittest` to test if all data can be load by Dataset_Manager

### Finetune AlexNet
By running `finetune.py`, the training of the last layer of AlexNet and the finetuning of all its layers are beginning. You now just have to wait, depending on your configuration



