# 3rd Party Packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.backend import image_data_format
from tensorflow.keras.regularizers import l2, l1
# User Packages
from .base import NeuralNetwork

class ShallowNet(NeuralNetwork):
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

        # CONV => RELU
        model.add(Conv2D(
            input_shape=inputShape,
            filters=32,
            kernel_size=(3,3),
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
        # Softmax
        model.add(Flatten())
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

