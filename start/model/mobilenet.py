# Mobile nets of all types

# 3rd Party Packages
import numpy as np
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as GoogleMobileNetV2
# User Packages
from .base import NeuralNetwork, FineTuningMixin

# Google's MobileNetV2
class MobileNetV2(NeuralNetwork, FineTuningMixin):
    def __init__(self):
        super().__init__()

    @staticmethod
    def build(properties: dict):
        input_shape = (properties['width'], properties['height'], properties['channels'])
        standard_input_shapes = (
            (96, 96, 3),
            (128, 128, 3),
            (160, 160, 3),
            (192, 192, 3),
            (224, 224, 3)
        )
        transferrable_input_shapes = (
            (96, 96, 1),
            (128, 128, 1),
            (160, 160, 1),
            (192, 192, 1),
            (224, 224, 1)
        )
        isImageNet = properties['weights'].lower() == 'imagenet'
        isStandardDimensions = input_shape in standard_input_shapes
        if input_shape not in transferrable_input_shapes:
            raise ValueError('Input shape must be either RGB color or grayscale: {}'.format(input_shape))
        # Leverage Google's model for trained 'imagenet' weights
        mnv2 = GoogleMobileNetV2(
            input_shape=input_shape,
            alpha=1.4,
            depth_multiplier=1,
            include_top=False,
            weights=None,
            input_tensor=None,
            pooling='avg'
        )
        # Replace the bottom (i.e. input layers) with input_shape input layers
        # and copy over imagenet weights
        if isImageNet and not isStandardDimensions:
            print('[INFO] ImageNet weights w/ Non-Standard Input Dimensions, redefining MobileNetV2 head...')
            mnv2_imagenet = GoogleMobileNetV2(
                input_shape=input_shape[:-1]+(3,),
                alpha=1.4,
                depth_multiplier=1,
                include_top=False,
                weights='imagenet',
                input_tensor=None,
                pooling='avg'
            )

            # Transfer weights: mnv2_imagenet => mnv2
            mnv2 = FineTuningMixin.transfer_weights(mnv2_imagenet, mnv2)

            # When transferring the imagenet weights from the first Conv2D layer, we must
            # take care to change the number of channels from 3 to properties['channels']
            # Do this by averaging the channel weights, since the first layer usually learns edge
            # features, and most edges occur in all color channels simultaneously
            # Thanks: https://jhui.github.io/2017/03/16/CNN-Convolutional-neural-network/
            weights = mnv2_imagenet.get_layer(name='Conv1').get_weights()
            weights[0] = np.reshape(np.mean(weights[0], axis=2), (3, 3, properties['channels'], 48))
            mnv2.get_layer(name='Conv1').set_weights(weights)

            # NOTE: I did an extensive study to see which layers would need to be retrained on Flowers-17.
            # Block 12+ came out to be the best block to train. Specifically, depthwise_relu,
            # was the layer with the least validation loss and smoothest training curve to train up to (not including).
            mnv2 = FineTuningMixin.train_up_to(mnv2, 'block_12_depthwise_relu')

            # Don't need the temporary imagenet model anymore
            del mnv2_imagenet

        model = Sequential()
        model.add(mnv2)
        # MobileNetV2 uses a GAP layer which essentially "Flattens" the feature maps already, so no call to Flatten here
        try:
            dense_units = properties['dense_units']
        except KeyError:
            dense_units = 256
        try:
            dropout_rate = properties['dropout_rate']
        except KeyError:
            dropout_rate = 0.4
        try:
            regularization_strength = properties['regularization_strength']
        except KeyError:
            regularization_strength = 0.001

        model.add(Dense(
            units=dense_units,
            activation='relu',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform',
            kernel_regularizer=l2(0.001),
            bias_regularizer=l2(0.001),
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None
        ))
        model.add(Dropout(dropout_rate))
        model.add(Dense(
            units=len(properties['classes']),
            activation='softmax',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='glorot_uniform',
            kernel_regularizer=l2(0.001),
            bias_regularizer=l2(0.001),
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None
        ))
        return model

