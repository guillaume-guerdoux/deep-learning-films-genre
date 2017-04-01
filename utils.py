import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

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
            training_loss.append(line_list[5].replace(',', ''))
    return iterations, training_loss


def draw_training_mean_average_precision(logs):
    iterations = []
    # iteration_counter = 50
    iteration_counter = 100
    training_precision = []
    for line in logs:
        line_list = line.split()
        # print(line_list)
        if line_list[0] == "Global" and \
           line_list[1] == "Training" and \
           line_list[2] == "Accuracy" and \
           line_list[3] == "=":
            iterations.append(iteration_counter)
            iteration_counter += 100
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


def new_draw_validation_mean_average_precision(logs):
    iterations = []
    validation_precision = []
    for line in logs:
        line_list = line.split()
        print(line_list)
        if line_list[0] == "Iter" and \
           line_list[2] == "Global" and \
           line_list[3] == "Validation" and \
           line_list[4] == "Accuracy":
            iterations.append(line_list[1])
            validation_precision.append(line_list[6])
    return iterations, validation_precision


if __name__ == "__main__":
    '''with open('labels.json') as json_data:
        labels = json.load(json_data)
    # print(labels)
    print(list(reversed(sorted(get_genre_stats(labels).items(), key=lambda x:x[1]))))'''

    # TRAINING LOSS
    mse_with_data_aumentation_logs = open(
        "logs/logs_to_compare_MSE_SCE_data_augmentation/MSE_with_data_augmentation-logs.txt",
        'r'
    )
    mse_without_data_aumentation_logs = open(
        "logs/logs_to_compare_MSE_SCE_data_augmentation/MSE_without_data_augmentation-logs.txt",
        'r'
    )

    sce_with_data_aumentation_logs = open(
        "logs/logs_to_compare_MSE_SCE_data_augmentation/SCE_with_data_augmentation-logs.txt",
        'r'
    )
    sce_without_data_aumentation_logs = open(
        "logs/logs_to_compare_MSE_SCE_data_augmentation/SCE_without_data_augmentation-logs.txt",
        'r'
    )

    mse_with_data_augmentation_iterations, mse_with_data_augmentation_training_loss = \
        draw_training_loss(mse_with_data_aumentation_logs)
    mse_without_data_augmentation_iterations, mse_without_data_augmentation_training_loss = \
        draw_training_loss(mse_without_data_aumentation_logs)
    sce_with_data_augmentation_iterations, sce_with_data_augmentation_training_loss = \
        draw_training_loss(sce_with_data_aumentation_logs)
    sce_without_data_augmentation_iterations, sce_without_data_augmentation_training_loss = \
        draw_training_loss(sce_without_data_aumentation_logs)

    mse_fig, mse_ax = plt.subplots()
    mse_ax.plot(
            mse_with_data_augmentation_iterations,
            mse_with_data_augmentation_training_loss,
            'r:',
            label="Mean square error / Data augmentation"
        )
    mse_ax.plot(
            mse_without_data_augmentation_iterations,
            mse_without_data_augmentation_training_loss,
            'b:',
            label="Mean square error / No data augmentation"
        )
    legend = mse_ax.legend(loc='upper right', prop={'size':13})
    plt.show()

    sce_fig, sce_ax = plt.subplots()
    sce_ax.plot(
            sce_with_data_augmentation_iterations,
            sce_with_data_augmentation_training_loss,
            'r:',
            label="Sigmoid Cross Entropy / Data augmentation"
        )
    sce_ax.plot(
            sce_without_data_augmentation_iterations,
            sce_without_data_augmentation_training_loss,
            'b:',
            label="Sigmoid Cross Entropy / No data augmentation"
        )
    legend = sce_ax.legend(loc='upper center', prop={'size':12})
    plt.show()

    # Training
    mse_with_data_aumentation_logs = open(
        "logs/logs_to_compare_MSE_SCE_data_augmentation/MSE_with_data_augmentation-logs.txt",
        'r'
    )
    mse_without_data_aumentation_logs = open(
        "logs/logs_to_compare_MSE_SCE_data_augmentation/MSE_without_data_augmentation-logs.txt",
        'r'
    )

    sce_with_data_aumentation_logs = open(
        "logs/logs_to_compare_MSE_SCE_data_augmentation/SCE_with_data_augmentation-logs.txt",
        'r'
    )
    sce_without_data_aumentation_logs = open(
        "logs/logs_to_compare_MSE_SCE_data_augmentation/SCE_without_data_augmentation-logs.txt",
        'r'
    )

    mse_with_data_augmentation_iterations, mse_with_data_augmentation_training_map = \
        draw_training_mean_average_precision(mse_with_data_aumentation_logs)
    mse_without_data_augmentation_iterations, mse_without_data_augmentation_training_map = \
        draw_training_mean_average_precision(mse_without_data_aumentation_logs)
    sce_with_data_augmentation_iterations, sce_with_data_augmentation_training_map = \
        draw_training_mean_average_precision(sce_with_data_aumentation_logs)
    sce_without_data_augmentation_iterations, sce_without_data_augmentation_training_map = \
        draw_training_mean_average_precision(sce_without_data_aumentation_logs)
    print(len(sce_with_data_augmentation_iterations), len(sce_with_data_augmentation_training_map))
    fig, ax = plt.subplots()
    ax.plot(
            mse_with_data_augmentation_iterations,
            mse_with_data_augmentation_training_map,
            'r:',
            label="Mean square error / Data augmentation"
        )
    ax.plot(
            mse_without_data_augmentation_iterations,
            mse_without_data_augmentation_training_map,
            'b:',
            label="Mean square error / No data augmentation"
        )
    ax.plot(
            sce_with_data_augmentation_iterations,
            sce_with_data_augmentation_training_map,
            'g:',
            label="Sigmoid Cross Entropy / Data augmentation"
        )
    ax.plot(
            sce_without_data_augmentation_iterations,
            sce_without_data_augmentation_training_map,
            'y:',
            label="Sigmoid Cross Entropy / No data augmentation"
        )
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    legend = ax.legend(loc='lower right',  prop={'size':12} )
    plt.show()

    # Validation
    mse_with_data_aumentation_logs = open(
        "logs/logs_to_compare_MSE_SCE_data_augmentation/MSE_with_data_augmentation-logs.txt",
        'r'
    )
    mse_without_data_aumentation_logs = open(
        "logs/logs_to_compare_MSE_SCE_data_augmentation/MSE_without_data_augmentation-logs.txt",
        'r'
    )

    sce_with_data_aumentation_logs = open(
        "logs/logs_to_compare_MSE_SCE_data_augmentation/SCE_with_data_augmentation-logs.txt",
        'r'
    )
    sce_without_data_aumentation_logs = open(
        "logs/logs_to_compare_MSE_SCE_data_augmentation/SCE_without_data_augmentation-logs.txt",
        'r'
    )
    # Mean average precision
    mse_with_data_augmentation_iterations, mse_with_data_augmentation_validation_map = \
        draw_validation_mean_average_precision(mse_with_data_aumentation_logs)
    mse_without_data_augmentation_iterations, mse_without_data_augmentation_validation_map = \
        draw_validation_mean_average_precision(mse_without_data_aumentation_logs)
    sce_with_data_augmentation_iterations, sce_with_data_augmentation_validation_map = \
        draw_validation_mean_average_precision(sce_with_data_aumentation_logs)
    sce_without_data_augmentation_iterations, sce_without_data_augmentation_validation_map = \
        draw_validation_mean_average_precision(sce_without_data_aumentation_logs)
    fig, ax = plt.subplots()
    ax.plot(
            mse_with_data_augmentation_iterations,
            mse_with_data_augmentation_validation_map,
            'r:',
            label="Mean square error / Data augmentation"
        )
    ax.plot(
            mse_without_data_augmentation_iterations,
            mse_without_data_augmentation_validation_map,
            'b:',
            label="Mean square error / No data augmentation"
        )
    ax.plot(
            sce_with_data_augmentation_iterations,
            sce_with_data_augmentation_validation_map,
            'g:',
            label="Sigmoid Cross Entropy / Data augmentation"
        )
    ax.plot(
            sce_without_data_augmentation_iterations,
            sce_without_data_augmentation_validation_map,
            'y:',
            label="Sigmoid Cross Entropy / No data augmentation"
        )
    legend = ax.legend(loc='lower right')
    plt.show()

    # Training dropout
    dropout_05 = open(
        "logs/logs_to_compare_dropout_learning_rate/dropout_05-logs.txt",
        'r'
    )
    dropout_075 = open(
        "logs/logs_to_compare_dropout_learning_rate/dropout0_75-logs.txt",
        'r'
    )

    dropout_1 = open(
        "logs/logs_to_compare_dropout_learning_rate/dropout_1-logs.txt",
        'r'
    )
    dropout_05_iterations, dropout_05_training_map = \
        draw_training_mean_average_precision(dropout_05)
    dropout_075_iterations, dropout_075_training_map = \
        draw_training_mean_average_precision(dropout_075)
    dropout_1_iterations, dropout_1_training_map = \
        draw_training_mean_average_precision(dropout_1)

    fig, ax = plt.subplots()
    ax.plot(
            dropout_05_iterations,
            dropout_05_training_map,
            'r:',
            label="Dropout de 0,5"
        )
    ax.plot(
            dropout_075_iterations,
            dropout_075_training_map,
            'b:',
            label="Dropout de 0,75"
        )
    ax.plot(
            dropout_1_iterations,
            dropout_1_training_map,
            'g:',
            label="Dropout de 1"
        )
    legend = ax.legend(loc='lower right')
    plt.show()

    # Validation dropout
    dropout_05 = open(
        "logs/logs_to_compare_dropout_learning_rate/dropout_05-logs.txt",
        'r'
    )
    dropout_075 = open(
        "logs/logs_to_compare_dropout_learning_rate/dropout0_75-logs.txt",
        'r'
    )

    dropout_1 = open(
        "logs/logs_to_compare_dropout_learning_rate/dropout_1-logs.txt",
        'r'
    )
    dropout_05_iterations, dropout_05_validation_map = \
        new_draw_validation_mean_average_precision(dropout_05)
    dropout_075_iterations, dropout_075_validation_map = \
        new_draw_validation_mean_average_precision(dropout_075)
    dropout_1_iterations, dropout_1_validation_map = \
        new_draw_validation_mean_average_precision(dropout_1)

    print(dropout_05_validation_map)
    fig, ax = plt.subplots()
    ax.plot(
            dropout_05_iterations,
            dropout_05_validation_map,
            'r:',
            label="Dropout de 0,5"
        )
    ax.plot(
            dropout_075_iterations,
            dropout_075_validation_map,
            'b:',
            label="Dropout de 0,75"
        )
    ax.plot(
            dropout_1_iterations,
            dropout_1_validation_map,
            'g:',
            label="Dropout de 1"
        )
    legend = ax.legend(loc='lower right')
    plt.show()


    # Final model
    final_model_model = open(
        "logs/logs_final_model/final_model-logs.txt",
        'r'
    )
    iterations_training, training_map = \
        draw_training_mean_average_precision(final_model_model)
    final_model_model = open(
        "logs/logs_final_model/final_model-logs.txt",
        'r'
    )
    iterations_validation, validation_map = \
        new_draw_validation_mean_average_precision(final_model_model)

    print(training_map, validation_map)
    fig, ax = plt.subplots()
    ax.plot(
            iterations_training,
            training_map,
            'r:',
            label="Ensemble de train"
        )
    ax.plot(
            iterations_validation,
            validation_map,
            'b:',
            label="Ensemble de validation"
        )
    legend = ax.legend(loc='lower right', prop={'size':16})
    plt.show()
