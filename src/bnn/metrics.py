import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch

import bnn.random


def distribution(tensor: torch.Tensor | None) -> bnn.random.DISCRETE_DIST | None:
    if tensor is None:
        return None

    numel = torch.numel(tensor)
    values = torch.unique(tensor)

    distribution = []
    for value in values:
        prop = float(torch.sum(tensor == value) / numel)
        distribution.append(bnn.random.VALUE_PROB_PAIR(value=value.item(), probability=prop))

    return distribution


def distribution_plot(distribution: bnn.random.DISCRETE_DIST) -> np.array:
    FIG_NAME = 'dist'

    num_plots = len(distribution)
    ceil_sqrt_num_plots = np.ceil(np.sqrt(num_plots)).astype(int)

    fig, axs = plt.subplots(ceil_sqrt_num_plots, ceil_sqrt_num_plots, num=FIG_NAME)
    axs = np.array(axs).flatten()

    min_x, max_x = -1, 1
    width = 0.1
    for i, (ax, d) in enumerate(zip(axs, distribution + [None] * len(axs))):
        if d is None:
            ax.plot(np.linspace(min_x, max_x), np.linspace(0, 1), '-', color='red')
            ax.plot(np.linspace(max_x, min_x), np.linspace(0, 1), '-', color='red')
            ax.axis('off')

        else:
            ax.set_title(i)
            vals = [pair.value for pair in d]
            probs = [pair.probability for pair in d]

            if True or len(vals) > 3:
                ax.plot(vals, probs, 'o-')
            else:
                ax.bar(vals, probs, width=width)

            min_x = min(min_x, min(vals))
            max_x = max(max_x, max(vals))

    for ax in axs:
        # ax.set_xlim(min_x-width, max_x+width)
        ax.set_ylim(0, None)
        ax.grid()

    fig.tight_layout(pad=0.1)

    image = _fig_to_numpy(fig)
    plt.close(FIG_NAME)

    return image


def _fig_to_numpy(fig: matplotlib.figure.Figure) -> np.array:
    canvas = fig.canvas
    canvas.draw()
    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image = image_flat.reshape(*reversed(canvas.get_width_height()), 3)

    return image
