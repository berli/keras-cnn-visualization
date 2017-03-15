from unittest import TestCase

from cnnvis import GradientVis
from keras.layers import Convolution2D, Flatten, Dense
from keras.models import Sequential


class TestGradientVis(TestCase):

    @classmethod
    def setUpClass(cls):
        model = Sequential()
        model.add(Convolution2D(2, 2, 2, activation='relu', border_mode='same',
                                input_shape=(6, 6, 1)))
        model.add(Flatten())
        model.add(Dense(2, activation='linear'))
        model.compile(optimizer='RMSprop', loss='mse', metrics=['mean_squared_error'])

        cls.model = model
        cls.gradient_vis = GradientVis(model)

    def test_mean_neurons(self):
        results = list(self.gradient_vis.mean_neurons(self.model.layers))
        self.assertEqual(len(results), 3, 'Number of vis results should be 2 in the example network')

    def test_center_neuron(self):
        results = list(self.gradient_vis.center_neuron(self.model.layers))
        self.assertEqual(len(results), 3, 'Number of vis results should be 2 in the example network')

    def test_all_neurons(self):
        results = list(self.gradient_vis.all_neurons(self.model.layers))
        self.assertEqual(len(results), 74, 'Number of vis results should be 6 in the example network')
