#!/usr/bin/env python3
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


def basic_plot(raw, fig=None, axes=None, palette='hls'):
    """
    # Plot the raw data, simple plot.
    #
    # Input
    # =====
    # `raw`: Raw EFI input signal, expect shape (`n_readout`, `n_stimuli`).
    # `fig`, `axes`: Matplotlib figure and axes handlers; if `None`, a `fig`
    #                and an `axes` handlers will be created.
    # `palette`: Seaborn colour palette name to change plotting colour.
    #
    # Return
    # =====
    # Matplotlib figure and axes handlers.
    """
    n_readout, n_stimuli = raw.shape
    x = np.arange(n_readout) + 1

    # Just set some cool colour...
    c = sns.color_palette(palette, n_readout)

    if (fig is None) or (axes is None):
        fig, axes = plt.subplots(1, 1)

    for i in range(n_stimuli):
        axes.plot(x, raw[:, i], c=c[i])

    plt.axis([1, 16, 0, 2])
    return fig, axes


