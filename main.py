import pickle
from dataset_manager import DatasetManager

with open('training_set_dict.pickle', 'rb') as handle:
    training_dict = pickle.load(handle)
with open('test_set_dict.pickle', 'rb') as handle:
    test_dict = pickle.load(handle)

dataset_manager = DatasetManager(training_dict, test_dict)
print(len(dataset_manager.training_samples_list))
print(len(dataset_manager.test_samples_list))
dataset_manager.cur_train = 3780
dataset_manager.cur_test = 1015
for i in range(1000):
    images = dataset_manager.next_batch(50, "train")
    print(images.shape)
    print(dataset_manager.cur_train)
    assert images.shape == (50, 227, 227, 3)
    images = dataset_manager.next_batch(50, "test")
    print(images.shape)
    print(dataset_manager.cur_test)
    assert images.shape == (50, 227, 227, 3)
