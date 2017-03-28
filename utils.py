import numpy as np
import pickle
import json
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


if __name__ == "__main__":
    with open('labels.json') as json_data:
        labels = json.load(json_data)
    # print(labels)
    print(get_genre_stats(labels))
