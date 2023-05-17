"""
    This module contains some useful functions for the project.
"""

import matplotlib.pyplot as plt
import time
import os
from colorama import Back, Style


def timeit(function):
    """
    Decorator who prints the execution time of a function
    :param function: function to be executed
    :type function: function
    :return: function's return
    :rtype:function's return type
    """

    def timed(*args, **kw):
        ts = time.time()
        print('\nExecuting %r ' % function.__name__)
        result = function(*args, **kw)
        te = time.time()
        print('\n%r executed in %2.2f s' % (function.__name__, (te - ts)))
        return result

    return timed


def plot_model_loss(trained_model, fig_save_dir):
    # Getting statistics values
    acc = trained_model.history['accuracy']
    val_acc = trained_model.history['val_accuracy']
    loss = trained_model.history['loss']
    val_loss = trained_model.history['val_loss']
    epochs = range(1, len(acc) + 1)
    # Validation loss
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(os.path.join(fig_save_dir, "val_loss.png"))


def plot_model_accuracy(trained_model, fig_save_dir):
    # Getting model history
    acc = trained_model.history['accuracy']
    val_acc = trained_model.history['val_accuracy']
    loss = trained_model.history['loss']
    val_loss = trained_model.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Configuring plot
    plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(os.path.join(fig_save_dir, "val_acc.png"))

def plot_model_metrics(plot_list, fig_save_dir, plot_title):
    plt.figure()
    for label, values in plot_list.items():
        plt.plot(values, label=label)
    plt.title(plot_title)
    plt.legend()
    plt.savefig(os.path.join(fig_save_dir, plot_title + ".png"))

def save_model_to_h5(model, save_dir):
    model_name = model.name
    model.save(save_dir + model_name + ".h5")


def save_structure_in_json(model, json_path):
    """
    Save the model structure in a json file.

    :param model: keras model
    :type model: kerqs.Model
    :param json_path: path of the jsn file
    :type json_path: str
    """

    json_string = model.to_json()
    with open(json_path, "w+") as f:
        f.write(json_string)


def safe_folder_creation(folder_path):
    """
    Safely create folder and return the new path value if a change occurred.

    :param folder_path: proposed path for the folder that must be created
    :type folder_path: str
    :return: path of the created folder
    :rtype: str
    """
    # Boolean initialization
    folder = True

    while folder:
        # Check path validity
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            folder = False
        else:
            folder_path = input(Back.RED + 'Folder already exists : {}\n Please enter a new path !'.format(folder_path)
                                + Style.RESET_ALL)
    return folder_path
