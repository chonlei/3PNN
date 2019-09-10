#
# Quick diagnostic plots.
#
from __future__ import absolute_import, division
from __future__ import print_function, unicode_literals
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


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


def basic_plot_splitted(raw, fig=None, axes=None, c='C0', ls=''):
    """
    # Get the curvature of the EFI measurement, with given a parameteric form.
    #
    # Input
    # =====
    # `raw`: Raw EFI input signal, expect shape (`n_readout`, `n_stimuli`).
    # `fig`, `axes`: Matplotlib figure and axes handlers; if `None`, a `fig`
    #                and an `axes` handlers will be created.
    # `c`: Plotting colour.
    # `ls`: Matplotlib linestyle argument.
    #
    # Return
    # =====
    # Matplotlib figure and axes handlers.
    """
    n_readout, n_stimuli = raw.shape
    x = np.arange(n_readout) + 1

    if (fig is None) or (axes is None):
        fig, axes = plt.subplots(4, 4, figsize=(14, 10))

    for i in range(n_stimuli):
        ai, aj = i // 4, i % 4
        axes[ai, aj].plot(x, raw[:, i], c=c, marker='o', ls=ls)
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


def fitted_curves_splitted(p, func, fig=None, axes=None, c='C2', ls='-'):
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
    # `c`: Plotting colour.
    # `ls`: Matplotlib linestyle argument.
    #
    # Return
    # =====
    # Matplotlib figure and axes handlers.
    """
    n_stimuli = len(p)
    n_readout = n_stimuli  # assume it is the case
    x = np.arange(n_readout) + 1

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
            axes[ai, aj].plot(x_plot, y_plot, c=c, ls=ls)

        # Left
        if p[i][1] is not None:
            # Calculate
            x2 = np.arange(1, i + 1)
            y2 = func(x2, *p[i][1])
            # For plot
            x_plot = x2
            y_plot = y2[::-1]
            # And plot
            axes[ai, aj].plot(x_plot, y_plot, c=c, ls=ls)

        axes[ai, aj].set_xlim([1, 16])
        axes[ai, aj].set_ylim([0, 2])
        axes[ai, aj].set_xticks(range(1, 17))

    axes[-1, 1].text(1.05, -0.3, 'Electrode #', ha='center', va='center',
            transform=axes[-1, 1].transAxes)
    axes[1, 0].text(-0.25, -0.25, r'Transimpedence (k$\Omega$)', ha='center',
            va='center', transform=axes[1, 0].transAxes, rotation=90)

    return fig, axes


def parameters(rt, rl, fig=None, axes=None, c='C0', marker='o', ls='',
        label=''):
    """
    # Plot the parameters.
    #
    # Input
    # =====
    # `rt`: Transversal resistance parameters, last one is basel resistance.
    # `rl`: Longitudinal resistance parameters.
    # `fig`, `axes`: Matplotlib figure and axes handlers; if `None`, a `fig`
    #                and an `axes` handlers will be created.
    # `c`: Plotting colour.
    # `marker`: Matplotlib marker argument.
    # `ls`: Matplotlib linestyle argument.
    # `label`: Matplotlib label argument.
    #
    # Return
    # =====
    # Matplotlib figure and axes handlers.
    """
    n_readout = len(rt)
    assert(len(rt) == len(rl) + 1)  # last one in R_T is R_basel
    x = np.arange(n_readout) + 1

    if (fig is None) or (axes is None):
        fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    axes[0].plot(x, rt, marker=marker, c=c, ls=ls, label=label)
    axes[0].set_yscale('log')
    axes[0].set_ylabel(r'$R_T$ (k$\Omega$)')
    axes[1].plot(x[:-1], rl, marker=marker, c=c, ls=ls, label=label)
    axes[1].set_ylabel(r'$R_L$ (k$\Omega$)')
    axes[1].set_xlabel('Resistor index')

    axes[1].set_xlim([1, 16])
    axes[1].set_xticks(range(1, 17))

    return fig, axes


def sensitivity_analyse_splitted(x, y, fig=None, axes=None, c='C0', marker='o',
        ls='', label='', xylabels=None):
    """
    # Plot the feature sensitivity plot.
    #
    # Input
    # =====
    # `x`: An input/printing parameter (x-axis), with shape (`n_points`, ).
    # `y`: A feature (y-axis), with shape (`n_points`, `n_stimuli`).
    # `fig`, `axes`: Matplotlib figure and axes handlers; if `None`, a `fig`
    #                and an `axes` handlers will be created.
    # `c`: Plotting colour.
    # `marker`: Matplotlib marker argument.
    # `ls`: Matplotlib linestyle argument.
    # `label`: Matplotlib label argument.
    # `xylabels`: [`x_label`, `y_label`] for the plot.
    #
    # Return
    # =====
    # Matplotlib figure and axes handlers.
    """
    n_points, n_stimuli = y.shape
    assert(len(x) == n_points)

    if (fig is None) or (axes is None):
        fig, axes = plt.subplots(4, 4, figsize=(14, 10))

    for i in range(n_stimuli):

        ai, aj = i // 4, i % 4

        if any(np.isfinite(y[:, i])):
            axes[ai, aj].plot(x, y[:, i], c=c, ls=ls, marker=marker,
                    label=label)

    if xylabels is not None:
        axes[-1, 1].text(1.05, -0.3, xylabels[0], ha='center', va='center',
                transform=axes[-1, 1].transAxes)
        axes[1, 0].text(-0.25, -0.25, xylabels[1], ha='center', va='center',
                transform=axes[1, 0].transAxes, rotation=90)

    return fig, axes


