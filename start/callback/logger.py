# Loggers for training neural networks
# Serializers, Graphers, etc.

from tensorflow.keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

class TrainingLogger(BaseLogger):
    # NOTE: BaseLogger parameters stateful_metrics is an iterable of string names of metrics that
    # should *not* be averaged over an epoch
    def __init__(self, figure_path: Path, *args, json_path: Path=None, start_at: int=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.figure_path = figure_path
        self.json_path = json_path
        self.start_at = start_at

    def on_train_begin(self, logs: dict={}):
        # Initialize the history dictionary
        self.history = {}

        # If JSON file exists, load the training history
        if self.json_path is not None:
            if os.path.isfile(self.json_path):
                # Thanks: https://stackoverflow.com/questions/9644110/difference-between-parsing-a-text-file-in-r-and-rb-mode
                self.history = json.loads(open(self.json_path, 'r').read())
                # If a starting epoch was supplied, begin logging there
                if self.start_at > 0:
                    # Trim entries past the starting epoch
                    for key in self.history.keys():
                        self.history[key] = self.history[key][:self.start_at]

    def on_epoch_end(self, epoch: int, logs: dict={}):
        # Store logs in dictionary
        # By default, this contains 4 keys:
        #   * train_loss
        #   * train_acc
        #   * val_loss
        #   * val_acc
        for (key, val) in logs.items():
            # Thanks: https://www.tutorialspoint.com/python/dictionary_get.htm
            # Get key, otherwise []
            value_list = self.history.get(key, [])
            value_list.append(val)
            self.history[key] = value_list
        # Serialize if JSON file exists
        if self.json_path is not None:
            with open(self.json_path, 'w') as out:
                out.write(json.dumps(self.history))
        # Plot the values, if there are at least two epochs, epochs start at 0
        N = len(self.history['loss'])
        if N > 1:
            # Plot training/validation loss and accuracy
            x_labels = np.arange(0, N)
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(x_labels, self.history['loss'], label='train_loss')
            plt.plot(x_labels, self.history['val_loss'], label='val_loss')
            plt.plot(x_labels, self.history['acc'], label='train_acc')
            plt.plot(x_labels, self.history['val_acc'], label='val_acc')
            plt.title('Training Loss and Accuracy [Epoch {}]'.format(N))
            plt.xlabel('Epoch #')
            plt.ylabel('Loss/Accuracy')
            plt.legend()
            # Save the figure
            plt.savefig(self.figure_path)
            plt.close()

