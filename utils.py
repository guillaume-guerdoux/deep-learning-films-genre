import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import copy

from collections import defaultdict

from sklearn.metrics import average_precision_score


def mean_average_precision(outputs, labels):
    ''' Output shape : [nb_output, 26]
        Label shape : [nb_output, 26] '''
    average_precisions = []
    for index, output in enumerate(outputs):
        label = labels[index]
        average_precision = average_precision_score(label, output)
        average_precisions.append(average_precision)
    return(sum(average_precisions)/len(average_precisions))


def get_genre_stats(labels):
    general_dict = defaultdict(int)
    for genre_list in labels.values():
        for genre in genre_list:
            general_dict[genre] += 1
    genre_sum = sum(general_dict.values())
    percentage_dict = {}
    for genre, number in general_dict.items():
        percentage_dict[genre] = round((number/genre_sum)*100, 2)
    return percentage_dict


def draw_training_loss(logs):
    iterations = []
    training_loss = []
    for line in logs:
        line_list = line.split()
        if line_list[0] == "Iter" and \
           line_list[2] == "Training" and \
           line_list[3] == "Loss" and \
           line_list[4] == "=":
            iterations.append(line_list[1])
            training_loss.append(line_list[5])
    return iterations, training_loss


def draw_training_mean_average_precision(logs):
    print("ok")
    iterations = []
    iteration_counter = 50
    training_precision = []
    for line in logs:
        print(line)
        line_list = line.split()
        print(line_list)
        if line_list[0] == "Global" and \
           line_list[1] == "Training" and \
           line_list[2] == "Accuracy" and \
           line_list[3] == "=":
            iterations.append(iteration_counter)
            iteration_counter += 50
            training_precision.append(line_list[4])
    print(iterations)
    return iterations, training_precision


def draw_validation_mean_average_precision(logs):
    iterations = []
    validation_precision = []
    for line in logs:
        line_list = line.split()
        if line_list[0] == "Iter" and \
           line_list[2] == "Global" and \
           line_list[3] == "Testing" and \
           line_list[4] == "Accuracy":
            iterations.append(line_list[1])
            validation_precision.append(line_list[6])
    return iterations, validation_precision

if __name__ == "__main__":
    '''with open('    labels.json') as json_data:
        labels = json.load(json_data)
    # print(labels)
    print(list(reversed(sorted(get_genre_stats(labels).items(), key=lambda x:x[1]))))'''
    logs = open("logs/logs_to_compare_MSE_SCE_data_augmentation/MSE_without_data_augmentation-logs.txt", 'r')
    iterations, validation_precision = draw_validation_mean_average_precision(logs)
    logs = open("logs/logs_to_compare_MSE_SCE_data_augmentation/MSE_without_data_augmentation-logs.txt", 'r')
    iterations, training_precision = draw_training_mean_average_precision(logs)
    print(len(iterations), len(validation_precision), len(training_precision))
    print(iterations)
    plt.plot(iterations, validation_precision, 'r--',
             iterations, training_precision, 'b--')
    plt.show()
