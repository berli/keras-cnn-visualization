from itertools import groupby
from typing import Iterable

from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import withStroke

import keras
import matplotlib.pyplot as plt
import numpy as np
import math

from mpl_toolkits.axes_grid import ImageGrid


class VisResult:

    def __init__(self, derived_input: np.ndarray, layer: keras.layers.Layer, filter_index: int):
        self.derived_input = derived_input
        self.layer = layer
        self.filter_index = filter_index

    def _clip(self):
        orig = self.derived_input
        normalized = (orig - orig.mean()) / (orig.std() + 1e-5) * 0.1
        clipped = np.clip(normalized + 0.5, 0, 1)
        return clipped

    def make_valid_image(self):
        if not self.is_non_zero_size:
            return self
        clipped = self._clip()
        if len(clipped.shape) == 3 and clipped.shape[2] == 1:
            clipped = clipped.squeeze(axis=2)
        valid_image = (clipped * 255).astype(np.uint8)
        self.derived_input = valid_image
        return self

    def reduce_channel(self, slice_object):
        if self.is_non_zero_size:
            self.derived_input = self.derived_input[..., slice_object]
        return self

    @property
    def is_non_zero_size(self) -> bool:
        return all(dim > 0 for dim in self.derived_input.shape)

    @property
    def filter_name(self):
        if self.filter_index == -1:
            return '?'
        return self.filter_index

    @property
    def title(self):
        return 'Layer {}, Filter {}'.format(self.layer.name, self.filter_name)

    def plot_image(self, **kwargs):
        plt.title(self.title)
        plt.imshow(self.derived_input, **kwargs)
        plt.show()
        return self


class ImageGridPlot:

    def __init__(self, cols=10):
        self.cols = cols

    @staticmethod
    def _add_inner_title(ax, title, loc, size=None, **kwargs):
        if size is None:
            size = dict(size=plt.rcParams['legend.fontsize'])
        at = AnchoredText(title, loc=loc, prop=size,
                          pad=0., borderpad=0.5,
                          frameon=False, **kwargs)
        ax.add_artist(at)
        at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
        return at

    def plot(self, results: Iterable[VisResult], **kwargs):
        for layer, layer_results in groupby(results, lambda r: r.layer):
            layer_results = list(layer_results)
            cols = self.cols
            rows = math.ceil(len(layer_results) / cols)
            fig = plt.figure(1, figsize=(2 * cols, 2 * rows))
            fig.suptitle('Layer {}'.format(layer.name), y=0.94, fontsize=18)
            grid = ImageGrid(fig, 111, (rows, cols), axes_pad=0)
            for i, result in enumerate(layer_results):
                self._add_inner_title(grid[i], result.filter_name, loc=3)
                if result.is_non_zero_size:
                    grid[i].imshow(result.derived_input, **kwargs)
            plt.show()
