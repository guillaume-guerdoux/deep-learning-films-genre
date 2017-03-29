import os
import random
import pickle


# TODO : Validation set too
''' This function creates two sets :
    - training_set : name of training movies
    - test_set : name of tests movies
'''
training_set = set()
validation_set = set()
test_set = set()

absolute_path = 'assets/posters/'
for image in os.listdir(absolute_path):
    prob = random.random()
    if prob >= 0.35:
        training_set.add(image)
    elif prob >= 0.15:
        validation_set.add(image)
    else:
        test_set.add(image)


with open('training_set_list.pickle', 'wb') as handle:
    pickle.dump(training_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('validation_set_list.pickle', 'wb') as handle:
    pickle.dump(validation_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('test_set_list.pickle', 'wb') as handle:
    pickle.dump(test_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
