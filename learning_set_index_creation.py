import os
import random
import pickle


# For the moment, only main images in set : TODO : Put processed images too
# TODO : Validation set too

training_dict = {}
test_dict = {}

absolute_path = 'assets/posters/'
i = 0
for folder in os.listdir(absolute_path):
    i += 1
    # for filename in os.listdir(absolute_path + folder):
    image_path = absolute_path + folder + '/' + folder + '.jpg'
    train_prob = random.random()
    if train_prob >= 0.2:
        training_dict[folder + '.jpg'] = folder
    else:
        test_dict[folder + '.jpg'] = folder

with open('training_set_dict.pickle', 'wb') as handle:
    pickle.dump(training_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('test_set_dict.pickle', 'wb') as handle:
    pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''print(len(training_dict))
print(len(test_dict))
LIST_KEY = training_dict.keys()
for key in training_dict.keys():
    print(key)
    break
LIST_KEY = training_dict.keys()
for key in training_dict.keys():
    print(key)
    break'''
'''with open('training_set_dict.pickle', 'rb') as handle:
    b = pickle.load(handle)
print(b)'''
