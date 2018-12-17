# Base class for neural networks

from abc import ABC, abstractmethod

class NeuralNetwork(ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def build(properties: dict):
        pass

class FineTuningMixin(object):
    @staticmethod
    def train_up_to(model, layer):
        """
        Freezes layers before specified layer for training
        :param model: Keras model to configure trainable layers for
        :param layer: str or int representing the layer name or layer number, respectively
        :return Keras model with configured trainable layers
        """
        if isinstance(layer, int):
            for l in model.layers[:layer]:
                l.trainable = False
        elif isinstance(layer, str):
            for l in model.layers:
                if l.get_config()['name'] == layer:
                   break
                else:
                    l.trainable = False
        else:
            raise ValueError('layer must be either a layer number (int) or name(str)')

        return model

    @staticmethod
    def transfer_weights(source_model, destination_model):
        """
        Attempts to transfer weights from one model to another, matched by layer name
        :param source_model: Keras model with source weights
        :param destination_model: Keras model to transfer source weights to
        :return: Keras destination model with source model weights
        """
        for layer in source_model.layers:
            try:
                weights = source_model.get_layer(name=layer.get_config()['name']).get_weights()
                destination_model.get_layer(name=layer.get_config()['name']).set_weights(weights)
            except:
                # Just transfer what you can
                pass

        return destination_model

