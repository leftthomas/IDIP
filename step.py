import matplotlib.pyplot as plt
import numpy as np

x = np.array([1, 3, 5, 7, 9])

mil_100 = np.array([36.3, 36.7, 36.7, 36.5, 36.6])
mil_300 = np.array([39.4, 39.5, 39.5, 39.6, 39.5])
mil_500 = np.array([40.2, 40.4, 40.3, 40.3, 40.3])

plt.figure(figsize=(10, 5))
plt.plot(x, mil_100, color='red', linewidth=1.0, label='100', marker='o', markerfacecolor='white', markersize=5)
for a, b in zip(x, mil_100):
    plt.text(a, b - 0.2, str(b), color='red')
plt.plot(x, mil_300, color='blue', linewidth=1.0, label='300', marker='D', markerfacecolor='white', markersize=4)
for a, b in zip(x, mil_300):
    plt.text(a, b - 0.2, str(b), color='blue')
plt.plot(x, mil_500, color='green', linewidth=1.0, label='500', marker='*', markerfacecolor='white')
for a, b in zip(x, mil_500):
    plt.text(a, b - 0.2, str(b), color='green')

plt.xticks(x)
plt.xlim((0.9, 9.1))
plt.ylim(bottom=36)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_xlabel('number of steps')
ax.set_ylabel('AP (%)')
plt.legend(loc='best')
# plt.show()

plt.savefig('results/step.pdf', bbox_inches='tight', pad_inches=0.1)