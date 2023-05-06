import bisect

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

records = {
    # Name, Memory, Epoch, AP
    'Mask R-CNN': [48.0, 12, 34.4],
    'TensorMask': [77.4, 72, 35.4],
    'SOLO': [64.0, 72, 36.8],
    'K-Net': [64.1, 36, 38.4],
    'QueryInst': [84.1, 36, 40.6],
    'DiffusionInst': [153.0, 60, 37.1],
    'FastInst': [51.9, 50, 38.6],
    'Ours': [61.7, 12, 40.2]
}

text_cfg = dict(color='dimgray', size=10)
memory_legend_cfg = dict(c='gainsboro', edgecolors='white', linewidths=1.5)
dpi, textsize, fig_width, fig_height = 192, 12, 1920, 1084

plt.rcParams.update({
    'legend.fontsize': 'large',
    'figure.figsize': (20, 8),
    'axes.labelsize': textsize,
    'axes.titlesize': textsize,
    'xtick.labelsize': textsize * 0.85,
    'ytick.labelsize': textsize * 0.85,
    'axes.titlepad': 25
})

ann_offsets = {
    'Mask R-CNN': (8, 10),
    'TensorMask': (0, 18),
    'SOLO': (4, 12),
    'K-Net': (2, 14),
    'QueryInst': (5, 18),
    'DiffusionInst': (2, 48),
    'FastInst': (4, 11),
    'Ours': (2, 14)
}


def get_radii(memory):
    # NOTE: memories range from 48 to 153 (ours: 62)
    boundaries = [
        (0, 0),
        (60, 12),  # slope: 0.2
        (100, 26),  # slope: 0.35
        (130, 38),  # slope: 0.4
        (160, 53)  # slope: 0.5
    ]

    x_boundaries = [it[0] for it in boundaries]
    y_boundaries = [it[1] for it in boundaries]

    # find out which interval the value lies in
    interval_idx = bisect.bisect_left(x_boundaries, memory) - 1

    # linear interpolation
    slope = [
        (p1[1] - p2[1]) / (p1[0] - p2[0])
        for (p1, p2) in zip(boundaries, boundaries[1:])
    ]
    y_offset = slope[interval_idx] * (memory - x_boundaries[interval_idx])

    radii = y_offset + y_boundaries[interval_idx]
    return radii


def get_area(memory):
    # area of bounding box
    return 4 * get_radii(memory) ** 2


# def add_annotation(ax, xs, ys, texts, offsets, label_formatter=None):
#     for x, y, t, offset in zip(xs, ys, texts, offsets):
#         formatter = label_formatter or str
#         label = formatter(t)
#
#         ax.annotate(
#             label,  # this is the text
#             (x, y),  # these are the coordinates to position the label
#             textcoords="offset points",  # how to position the text
#             xytext=offset,  # distance from text to points (x,y)
#             ha='center',  # horizontal alignment
#             fontsize=textsize,  # NOTE: an ugly workaround
#         )


names = [it for it in records.keys()]
memories = [it[0] for it in records.values()]
epochs = [it[1] for it in records.values()]
aps = [it[2] for it in records.values()]
areas = [get_area(it) for it in memories]

colors = ListedColormap(
    ['lightpink', 'cornflowerblue', 'sandybrown', 'paleturquoise', 'plum', 'bisque', 'mediumpurple', 'red'])
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(fig_width / dpi, fig_height / dpi), width_ratios=(1, 4.5))
fig.subplots_adjust(wspace=0.05)  # adjust space between axes

for ax in axes:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_axisbelow(True)

    # plot the same data on both axes
    ax.grid(c='silver', linestyle='solid', which='major', alpha=0.5)
    # loosely dotted line, ref: https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
    ax.grid(c='silver', linestyle=(0, (1, 3)), which='minor', alpha=0.4)  # (offset, on-off-seq)

    scatter = ax.scatter(
        x=epochs,
        y=aps,
        s=areas,
        c=list(range(len(records))),
        cmap=colors,
        edgecolors='white',
        linewidths=1.5)
    # draw concentric circles at each scatter
    ax.scatter(x=epochs, y=aps, s=4, c='white', linewidths=0)

    # add annotation
    # add_annotation(
    #     ax=ax,
    #     xs=epochs,
    #     ys=aps,
    #     texts=aps,
    #     label_formatter=lambda ap: f'{ap}',
    #     offsets=[ann_offsets[it] for it in names])

handle, _ = scatter.legend_elements(prop='colors')
axes[1].legend(handles=handle, labels=names, loc="upper right", ncols=2, markerscale=1.2)

# hide the spines between ax1 and ax2
axes[0].spines['right'].set_visible(False)
axes[1].spines['left'].set_visible(False)

# since sharey=True, this affects both axes
axes[0].set_ylim(31.2, 41.8)

axes[0].set_xlim(0, 16)
axes[0].set_xticks(np.arange(0, 13, 12))
axes[0].set_xticks(np.arange(0, 17, 4), minor=True)
axes[0].set_yticks(np.arange(32, 41.8, 1))
axes[0].yaxis.tick_left()

axes[1].set_xlim(32, 86)
axes[1].set_xticks(np.arange(36, 86, 12))
axes[1].set_xticks(np.arange(32, 86, 4), minor=True)
axes[1].tick_params(length=0)
axes[1].yaxis.tick_right()

d = 0.1  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(
    marker=[(-d, -d), (d, d)],
    markersize=12,
    linestyle="none",
    color='k',
    clip_on=False,
)
axes[0].plot([1], 0, transform=axes[0].transAxes, **kwargs)
axes[1].plot([0], 0, transform=axes[1].transAxes, **kwargs)

axes[0].set_ylabel('AP (%)')

# A tricky way to set common xlabel for two subplots, ref: https://stackoverflow.com/a/50610853
axes[0].set_xlabel('.', color=(0, 0, 0, 0))  # Reserve space for axis labels
fig.text(0.5, 0.04, 'number of epochs', va='center', ha='center', fontsize=plt.rcParams['axes.labelsize'])

# plot legends of memories
axes[1].scatter(
    x=[60, 62.5, 66.5, 72, 80],
    y=[32.9] * 5,
    s=[get_area(it) for it in [30, 60, 100, 130, 160]],
    **memory_legend_cfg
)

axes[1].text(59.3, 34.5, s='30G', **text_cfg)
axes[1].text(61.6, 34.5, s='60G', **text_cfg)
axes[1].text(65.4, 34.5, s='100G', **text_cfg)
axes[1].text(70.8, 34.5, s='130G', **text_cfg)
axes[1].text(78.9, 34.5, s='160G', **text_cfg)

plt.tight_layout()
plt.savefig('results/comp.pdf', dpi=dpi)