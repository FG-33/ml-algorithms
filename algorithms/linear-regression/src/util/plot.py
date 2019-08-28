import matplotlib.pyplot as plt
from matplotlib import interactive


def plot(num_fig, ia, labels, data, symbols):
    plt.figure(num_fig)
    interactive(ia)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    for idx, d in enumerate(data):
        plt.plot(d[0], d[1], symbols[idx])

    plt.show()
