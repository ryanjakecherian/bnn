import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import torch

import bnn.random



# two histrogram makers: one for integer/floats where #bins = range/std. dev., one for ternary which only has 3 values

def integer_or_float_hist(input: torch.Tensor):
    dist = []
    std = input.std().item()
    
    if std == 0:
        dist.append(bnn.random.VALUE_PROB_PAIR(value=input.mean().item(), probability=1))
        return dist
    
    bins = int((input.max().item() - input.min().item())/std)

    hist_counts = torch.histc(input, bins=bins, min=input.min().item(), max=input.max().item()) 
    bin_edges = torch.linspace(input.min().item(), input.max().item(), bins + 1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2  # midpoints of bins

    dict = {mid.item(): count.item() for mid, count in zip(bin_mids, hist_counts)}
    
    for i in range(len(dict)):
        key = list(dict.keys())[i]
        hist_value = dict[key]
        dist.append(bnn.random.VALUE_PROB_PAIR(value=key, probability=hist_value/input.numel()))    #if we want absolute number of elements, just remove the division by numel
    return dist


def tern_hist(tensor: torch.Tensor | None) -> bnn.random.DISCRETE_DIST | None:
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
            ax.plot(np.linspace(min_x, max_x), np.linspace(0, 1), '-', color='white')   #set colour to white instead of red (joao + red cross = no happy)
            ax.plot(np.linspace(max_x, min_x), np.linspace(0, 1), '-', color='white')
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