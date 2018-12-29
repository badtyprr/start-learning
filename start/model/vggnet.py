# 3rd Party Packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.backend import image_data_format
from tensorflow.keras.regularizers import l2, l1
# User Packages
from .base import NeuralNetwork

class MiniVGGNet(NeuralNetwork):
    def __init__(self):
        super().__init__()

    @staticmethod
    def build(properties: dict):
        """
        Builds a MiniVGGNet model
        :param properties: dict type that represents the model parameters:
            'width': int type representing the horizontal resolution
            'height': int type representing the vertical resolution
            'channels': int type representing the number of color channels, 1 for monochrome
            'classes': int type representing the number of classes to predict
        :return:
        """
        model = Sequential()
        # Align dimensions depending on channels_first or last
        channelDimension = -1   # end of the shape tuple
        if image_data_format() == "channels_first":
            inputShape = (
                properties['channels'],
                properties['height'],
                properties['width']
            )
            channelDimension = 1    # right after the batch index
        else:
            inputShape = (
                properties['height'],
                properties['width'],
                properties['channels']
            )

        # Construct model iteratively
        for j in range(2):
            for i in range(2):
                # CONV(i,j) => RELU(i,j)
                model.add(Conv2D(
                    input_shape=inputShape,
                    filters=32,
                    kernel_size=(3, 3),
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
                # BN(i,j)
                model.add(BatchNormalization(
                    axis=channelDimension,
                    momentum=0.99,
                    epsilon=0.001,
                    center=True,
                    scale=True,
                    beta_initializer='zeros',
                    gamma_initializer='ones',
                    moving_mean_initializer='zeros',
                    moving_variance_initializer='ones',
                    beta_regularizer=None,
                    gamma_regularizer=None,
                    beta_constraint=None,
                    gamma_constraint=None
                ))
            # POOL(j)
            model.add(MaxPooling2D(
                pool_size=(2, 2),
                strides=None,
                padding="same",
                data_format=None
            ))
            # DROPOUT(j)
            model.add(Dropout(
                rate=0.25,
                noise_shape=None,
                seed=None
            ))


        # FC1
        model.add(Flatten())
        model.add(Dense(
            units=512,
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
        # BN-FC1
        model.add(BatchNormalization(
            axis=channelDimension,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer='zeros',
            gamma_initializer='ones',
            moving_mean_initializer='zeros',
            moving_variance_initializer='ones',
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None,
            gamma_constraint=None
        ))
        # DROPOUT-FC1
        model.add(Dropout(
            rate=0.5,
            noise_shape=None,
            seed=None
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

