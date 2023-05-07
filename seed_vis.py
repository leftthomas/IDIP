import matplotlib.pyplot as plt
import numpy as np

# raw data
records = {
    'ResNet50-100': [36.6, 36.6, 36.7, 36.6, 36.7, 36.7, 36.4, 36.9, 36.6, 36.7],
    'ResNet50-300': [39.6, 39.5, 39.5, 39.5, 39.5, 39.5, 39.5, 39.4, 39.4, 39.5],
    'ResNet50-500': [40.3, 40.2, 40.2, 40.3, 40.3, 40.4, 40.2, 40.3, 40.3, 40.4],
}

# create figure
plt.style.use('_mpl-gallery')
dpi, fig_width, fig_height = 192, 960, 960

fig, ax = plt.subplots(figsize=(fig_width / dpi, fig_height / dpi))
plt.gca().xaxis.grid(False)
plt.gca().yaxis.grid(True)

vp = ax.violinplot(
    dataset=records.values(),
    vert=True,
    showmedians=True,
    showextrema=True,
)
ax.set_ylabel('AP (%)')
ax.set_xlabel('number of boxes')
ax.set_xticks(np.arange(5), labels=['', '100', '300', '500', ''])
ax.set_ylim(36.2, 41)
ax.set_yticks([37, 38, 39, 40, 41])

# styling
for body in vp['bodies']:
    body.set_alpha(0.6)

# calculate number of obs per group & median to position labels
medians = [np.median(it) for it in records.values()]
# add text to the figure
for i, v in enumerate(medians):
    plt.text((i + 1.40), (v), str(round(v, 1)), horizontalalignment='center', size='small', verticalalignment='center')

plt.tight_layout()
plt.savefig('results/seed.pdf', dpi=dpi)