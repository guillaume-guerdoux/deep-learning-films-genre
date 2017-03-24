import numpy as np
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
