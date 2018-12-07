# Keras implementation of LeNet (1998)

# 3rd Party Packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.backend import image_data_format
# User Packages
from .base import NeuralNetwork

class LeNet(NeuralNetwork):
    def __init__(self):
        super().__init__()

    @staticmethod
    def build(properties: dict):
        model = Sequential()

        # Align dimensions depending on channels_first or last
        if image_data_format() == "channels_first":
            inputShape = (
                properties['channels'],
                properties['height'],
                properties['width']
            )
        else:
            inputShape = (
                properties['height'],
                properties['width'],
                properties['channels']
            )

        # CONV1 => RELU1
        model.add(Conv2D(
            input_shape=inputShape,
            filters=20,
            kernel_size=(5, 5),
            padding="same",
            strides=1,
            dilation_rate=1,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None
        ))
        # POOL1
        model.add(MaxPooling2D(
            pool_size=(2, 2),
            strides=None,
            padding="same",
            data_format=None
        ))
        # CONV2 => RELU2
        model.add(Conv2D(
            input_shape=inputShape,
            filters=50,
            kernel_size=(5, 5),
            padding="same",
            strides=1,
            dilation_rate=1,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None
        ))
        # POOL2
        model.add(MaxPooling2D(
            pool_size=(2, 2),
            strides=None,
            padding="same",
            data_format=None
        ))
        # FC1
        model.add(Flatten())
        model.add(Dense(
            units=500,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None
        ))
        # FC2
        model.add(Dense(
            units=properties['classes'],
            activation='softmax',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform',
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None
        ))

        return model