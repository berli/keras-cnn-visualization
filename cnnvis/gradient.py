from typing import Iterable, Callable, Tuple, Dict
from keras import backend as K

import keras
import numpy as np
import itertools

from keras.engine import InputLayer
from keras.layers import Dropout, Flatten

import cnnvis
from cnnvis.visualize import VisResult
from keras.layers import Layer


class GradientVis:

    def __init__(self, model: keras.models.Model, iterations=20, learning_rate=1.0,
                 input_shape: Tuple=None):
        self.model = model
        self.iterations = iterations
        self.learning_rate = learning_rate

        if input_shape is None:
            self.input_shape = (1,) + self.model.input_shape[1:]
        else:
            self.input_shape = input_shape

        self.output_shapes = self._infer_output_shapes(self.input_shape)

    def _infer_output_shapes(self, input_shape: Tuple) -> Dict[Layer, Tuple]:
        output_shapes = {}
        for layer in self.model.layers:
            if isinstance(layer, InputLayer):
                continue
            output_shape = layer.compute_output_shape(input_shape)
            output_shapes[layer] = output_shape
            input_shape = output_shape
        return output_shapes

    @staticmethod
    def _is_conv_layer(layer: Layer):
        return len(layer.output_shape) >= 3

    @staticmethod
    def _should_skip_layer(layer: Layer):
        return isinstance(layer, Dropout) or isinstance(layer, Flatten) or isinstance(layer, InputLayer)

    def _filter_valid_layers(self, layers: Iterable[Layer]) -> Iterable[Layer]:
        return (layer for layer in layers if not self._should_skip_layer(layer))

    def _evaluate_neurons(self, layers: Iterable[Layer], ascent: bool,
                          neurons_for_layer: Callable[[Layer, int], Iterable]):

        for layer in self._filter_valid_layers(layers):
            if self._is_conv_layer(layer):
                for filter_index in range(layer.output_shape[-1]):
                    neurons = neurons_for_layer(layer, filter_index)
                    for neuron in neurons:
                        derived_input = self._derive_input(neuron, ascent)
                        yield VisResult(derived_input, layer, filter_index)
            else:
                neurons = neurons_for_layer(layer, -1)
                for neuron in neurons:
                    derived_input = self._derive_input(neuron, ascent)
                    yield VisResult(derived_input, layer, -1)

    def mean_neurons(self, layers: Iterable[Layer], ascent=True):

        def mean_for_layer(layer: Layer, filter_index: int):
            if filter_index != -1:
                return [K.mean(layer.output[..., filter_index])]
            return [K.mean(layer.output)]

        return self._evaluate_neurons(layers, ascent, mean_for_layer)

    def center_neuron(self, layers: Iterable[Layer], ascent=True):

        def center_for_layer(layer: Layer, filter_index: int):
            neuron = np.array(self.output_shapes[layer])
            neuron[0] = 0
            neuron[1:] //= 2
            if filter_index != -1:
                neuron[-1] = filter_index
            return [layer.output[tuple(neuron)]]

        return self._evaluate_neurons(layers, ascent, center_for_layer)

    def all_neurons(self, layers: Iterable[Layer], ascent=True):

        def all_for_layer(layer: Layer, filter_index: int):
            output_shape = self.output_shapes[layer]
            if filter_index != -1:
                output_shape = output_shape[:-1]
            for neuron in itertools.product(*(range(d) for d in output_shape)):
                if filter_index != -1:
                    neuron = tuple(neuron) + (filter_index,)
                yield layer.output[tuple(neuron)]

        return self._evaluate_neurons(layers, ascent, all_for_layer)


    @staticmethod
    def _get_receptive_field(random: np.ndarray, derived: np.ndarray) -> np.ndarray:
        mask = random != derived
        bbox = cnnvis.utils.bbox(mask)
        slicing = [slice(b_min, b_max) for b_min, b_max in cnnvis.utils.pairwise(bbox)]
        slicing.reverse()
        return derived[slicing]

    def _derive_input(self, loss, ascent: bool):
        input_tensor = self.model.input

        # Configure learning phase (e.g. disables dropout)
        learning_phase = K.learning_phase()
        learning_phase_value = 0

        # Compute the gradient of the input w.r.t. the given loss
        grads = K.gradients(loss, input_tensor)[0]

        # Normalization trick
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # Define function that returns the loss and gradients given the input tensor
        iterate = K.function([input_tensor, learning_phase], [loss, grads])

        # Define noise input
        random_input = np.random.random(self.input_shape) * 0.5 + 0.5
        derived_input = np.copy(random_input)

        # Run gradient ascent / descent
        for i in range(self.iterations):
            loss_value, grads_value = iterate([derived_input, learning_phase_value])
            if ascent:
                derived_input += grads_value * self.learning_rate
            else:
                derived_input -= grads_value * self.learning_rate

        # Cut out receptive field (one-element batch)
        receptive_field = self._get_receptive_field(random_input[0], derived_input[0])

        return receptive_field
