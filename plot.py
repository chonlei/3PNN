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

    axes.set_xlim([1, 16])
    axes.set_ylim([0, 2])
    axes.set_xticks(range(1, 17))
    axes.set_xlabel('Electrode #')
    axes.set_ylabel(r'Transimpedence (k$\Omega$)')
    return fig, axes


def basic_plot_splitted(raw, fig=None, axes=None, color='C0'):
    """
    # Get the curvature of the EFI measurement, with given a parameteric form.
    #
    # Input
    # =====
    # `raw`: Raw EFI input signal, expect shape (`n_readout`, `n_stimuli`).
    # `fig`, `axes`: Matplotlib figure and axes handlers; if `None`, a `fig`
    #                and an `axes` handlers will be created.
    # `color`: Plotting colour.
    #
    # Return
    # =====
    # Matplotlib figure and axes handlers.
    """
    n_readout, n_stimuli = raw.shape
    x = np.arange(n_readout) + 1

    # Just set some cool colour...
    c = color

    if (fig is None) or (axes is None):
        fig, axes = plt.subplots(4, 4, figsize=(14, 10))

    for i in range(n_stimuli):
        ai, aj = i // 4, i % 4
        axes[ai, aj].scatter(x, raw[:, i], color=c)
        axes[ai, aj].set_xlim([1, 16])
        axes[ai, aj].set_ylim([0, 2])
        axes[ai, aj].set_xticks(range(1, 17))

    axes[-1, 1].text(1.05, -0.3, 'Electrode #', ha='center', va='center',
            transform=axes[-1, 1].transAxes)
    axes[1, 0].text(-0.25, -0.25, r'Transimpedence (k$\Omega$)', ha='center',
            va='center', transform=axes[1, 0].transAxes, rotation=90)

    return fig, axes


def fitted_curves(p, func, fig=None, axes=None, palette='hls'):
    """
    # Get the curvature of the EFI measurement, with given a parameteric form.
    #
    # Input
    # =====
    # `p`: Parameters for `func`; expect a dictionary with the stimulation 
    #      electrode number as the key, and parameters as the value.
    # `func`: Function to fit to each curve, with `n_parameters`, giving the
    #         parameters as the gradients.
    # `fig`, `axes`: Matplotlib figure and axes handlers; if `None`, a `fig`
    #                and an `axes` handlers will be created.
    # `palette`: Seaborn colour palette name to change plotting colour.
    #
    # Return
    # =====
    # Matplotlib figure and axes handlers.
    """
    n_stimuli = len(p)
    n_readout = n_stimuli  # assume it is the case
    x = np.arange(n_readout) + 1

    # Just set some cool colour...
    c = sns.color_palette(palette, n_readout)

    if (fig is None) or (axes is None):
        fig, axes = plt.subplots(1, 1)

    for i in range(n_stimuli):

        # Right
        if p[i][0] is not None:
            # Calculate
            x1 = np.arange(1, n_readout - i - 1)
            y1 = func(x1, *p[i][0])
            # For plot
            x_plot = x1 + i + 1
            y_plot = y1
            # And plot
            axes.plot(x_plot, y_plot, c=c[i])

        # Left
        if p[i][1] is not None:
            # Calculate
            x2 = np.arange(1, i + 1)
            y2 = func(x2, *p[i][1])
            # For plot
            x_plot = x2
            y_plot = y2[::-1]
            # And plot
            axes.plot(x_plot, y_plot, c=c[i])

    axes.set_xlim([1, 16])
    axes.set_ylim([0, 2])
    axes.set_xticks(range(1, 17))
    axes.set_xlabel('Electrode #')
    axes.set_ylabel(r'Transimpedence (k$\Omega$)')

    return fig, axes


def fitted_curves_splitted(p, func, fig=None, axes=None, color='C2'):
    """
    # Get the curvature of the EFI measurement, with given a parameteric form.
    #
    # Input
    # =====
    # `p`: Parameters for `func`; expect a dictionary with the stimulation 
    #      electrode number as the key, and parameters as the value.
    # `func`: Function to fit to each curve, with `n_parameters`, giving the
    #         parameters as the gradients.
    # `fig`, `axes`: Matplotlib figure and axes handlers; if `None`, a `fig`
    #                and an `axes` handlers will be created.
    # `color`: Plotting colour.
    #
    # Return
    # =====
    # Matplotlib figure and axes handlers.
    """
    n_stimuli = len(p)
    n_readout = n_stimuli  # assume it is the case
    x = np.arange(n_readout) + 1

    # Just set some cool colour...
    c = color

    if (fig is None) or (axes is None):
        fig, axes = plt.subplots(4, 4, figsize=(14, 10))

    for i in range(n_stimuli):

        ai, aj = i // 4, i % 4

        # Right
        if p[i][0] is not None:
            # Calculate
            x1 = np.arange(1, n_readout - i - 1)
            y1 = func(x1, *p[i][0])
            # For plot
            x_plot = x1 + i + 1
            y_plot = y1
            # And plot
            axes[ai, aj].plot(x_plot, y_plot, c=c)

        # Left
        if p[i][1] is not None:
            # Calculate
            x2 = np.arange(1, i + 1)
            y2 = func(x2, *p[i][1])
            # For plot
            x_plot = x2
            y_plot = y2[::-1]
            # And plot
            axes[ai, aj].plot(x_plot, y_plot, c=c)

        axes[ai, aj].set_xlim([1, 16])
        axes[ai, aj].set_ylim([0, 2])
        axes[ai, aj].set_xticks(range(1, 17))

    axes[-1, 1].text(1.05, -0.3, 'Electrode #', ha='center', va='center',
            transform=axes[-1, 1].transAxes)
    axes[1, 0].text(-0.25, -0.25, r'Transimpedence (k$\Omega$)', ha='center',
            va='center', transform=axes[1, 0].transAxes, rotation=90)

    return fig, axes
