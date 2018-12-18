# Geometric Transformations (not provided by Keras)

from tensorflow.keras.layers import Layer
from tensorflow.keras.backend import shape, cast, expand_dims, clip, flatten, repeat_elements, int_shape, arange, \
    reshape, gather, ones_like, concatenate, stack
import numpy as np

class AffineTransformLayer(Layer):
    """
    Affine Transform from input transformation matrix
    References
    ----------
    [1]  Spatial Transformer Networks, Max Jaderberg, et al.
    [2]  https://github.com/skaae/transformer_network
    [3]  https://github.com/EderSantana/seya

    We recommend that descendants of Layer implement the following methods:

        __init__(): Save configuration in member variables
        build(): Called once from __call__, when we know the shapes of inputs and dtype. Should have the calls to add_weight(), and then call the super's build() (which sets self.built = True, which is nice in case the user wants to call build() manually before the first __call__).
        call(): Called in __call__ after making sure build() has been called once. Should actually perform the logic of applying the layer to the input tensors (which should be passed in as the first argument).
    """

    def __init__(self, output_size, **kwargs):
        self.output_size = output_size
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shapes):
        height, width = self.output_size
        num_channels = input_shapes[0][-1]
        return (None, height, width, num_channels)

    def call(self, tensors, mask=None):
        X, transformation = tensors
        output = self._transform(X, transformation, self.output_size)
        return output

    def _interpolate(self, image, sampled_grids, output_size):
        batch_size = shape(image)[0]
        height = shape(image)[1]
        width = shape(image)[2]
        num_channels = shape(image)[3]

        x = cast(flatten(sampled_grids[:, 0:1, :]), dtype='float32')
        y = cast(flatten(sampled_grids[:, 1:2, :]), dtype='float32')

        x = .5 * (x + 1.0) * cast(width, dtype='float32')
        y = .5 * (y + 1.0) * cast(height, dtype='float32')

        x0 = cast(x, 'int32')
        x1 = x0 + 1
        y0 = cast(y, 'int32')
        y1 = y0 + 1

        max_x = int(int_shape(image)[2] - 1)
        max_y = int(int_shape(image)[1] - 1)

        x0 = clip(x0, 0, max_x)
        x1 = clip(x1, 0, max_x)
        y0 = clip(y0, 0, max_y)
        y1 = clip(y1, 0, max_y)

        pixels_batch = arange(0, batch_size) * (height * width)
        pixels_batch = expand_dims(pixels_batch, axis=-1)
        flat_output_size = output_size[0] * output_size[1]
        base = repeat_elements(pixels_batch, flat_output_size, axis=1)
        base = flatten(base)

        # base_y0 = base + (y0 * width)
        base_y0 = y0 * width
        base_y0 = base + base_y0
        # base_y1 = base + (y1 * width)
        base_y1 = y1 * width
        base_y1 = base_y1 + base

        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = reshape(image, shape=(-1, num_channels))
        flat_image = cast(flat_image, dtype='float32')
        pixel_values_a = gather(flat_image, indices_a)
        pixel_values_b = gather(flat_image, indices_b)
        pixel_values_c = gather(flat_image, indices_c)
        pixel_values_d = gather(flat_image, indices_d)

        x0 = cast(x0, 'float32')
        x1 = cast(x1, 'float32')
        y0 = cast(y0, 'float32')
        y1 = cast(y1, 'float32')

        area_a = expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = expand_dims(((x - x0) * (y - y0)), 1)

        values_a = area_a * pixel_values_a
        values_b = area_b * pixel_values_b
        values_c = area_c * pixel_values_c
        values_d = area_d * pixel_values_d
        return values_a + values_b + values_c + values_d

    def _make_regular_grids(self, batch_size, height, width):
        # making a single regular grid
        x_linspace = np.linspace(-1., 1., width)
        y_linspace = np.linspace(-1., 1., height)
        x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
        x_coordinates = flatten(x_coordinates)
        y_coordinates = flatten(y_coordinates)
        ones = ones_like(x_coordinates)
        grid = concatenate([x_coordinates, y_coordinates, ones], 0)

        # repeating grids for each batch
        grid = flatten(grid)
        grids = np.tile(grid, stack([batch_size]))
        return reshape(grids, (batch_size, 3, height * width))

    def _transform(self, X, affine_transformation, output_size):
        batch_size, num_channels = shape(X)[0], shape(X)[3]
        transformations = reshape(affine_transformation,
                                    shape=(batch_size, 2, 3))
        # transformations = cast(affine_transformation[:, 0:2, :], 'float32')
        regular_grids = self._make_regular_grids(batch_size, *output_size)
        sampled_grids = K.batch_dot(transformations, regular_grids)
        interpolated_image = self._interpolate(X, sampled_grids, output_size)
        new_shape = (batch_size, output_size[0], output_size[1], num_channels)
        interpolated_image = reshape(interpolated_image, new_shape)
        return interpolated_image
