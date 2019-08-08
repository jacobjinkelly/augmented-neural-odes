"""
Parse the pickle files
"""
import pickle

import numpy as np
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt

dir = "img_results_2019-07-28_23-14_cifar10_experiments"
epochs = [0, 4, 9, 14, 19]
num_itrs = 196
batch_size = 256

for epoch in epochs:
    all_data = []
    for itr in range(num_itrs):
        with open("{}/epoch{}_itr{}_batch_nfe_history.pickle".format(dir, epoch, itr), "rb") as f:
            batch_nfe_history = pickle.load(f)

        with open("{}/epoch{}_itr{}_batch_timestamps_history.pickle".format(dir, epoch, itr), "rb") as f:
            batch_timestamps_history = pickle.load(f)

        with open("{}/epoch{}_itr{}_sep_nfe_history.pickle".format(dir, epoch, itr), "rb") as f:
            sep_nfe_history = pickle.load(f)

        with open("{}/epoch{}_itr{}_sep_timestamps_history.pickle".format(dir, epoch, itr), "rb") as f:
            sep_timestamps_history = pickle.load(f)

        batch_timestamps_history.sort()

        # sort timestamps lexicographically
        for i in range(len(sep_timestamps_history)):
            sep_timestamps_history[i].sort()
        sep_timestamps_history.sort()
        sep_timestamps_history.sort(key=len)

        sep_timestamps_history = np.array(sep_timestamps_history)
        timestamps_history_y = []

        colours = []
        COLOUR_SCALE = 1

        batch_timestamps_history = np.array(batch_timestamps_history)
        timestamps_history_y.append(np.repeat(-10, batch_timestamps_history.shape[0]))
        colours.extend([batch_timestamps_history.shape[0] / COLOUR_SCALE
                        for _ in range(batch_timestamps_history.shape[0])])

        for i in range(len(sep_timestamps_history)):
            sep_timestamps_history[i] = np.array(sep_timestamps_history[i])
            timestamps_history_y.append(np.repeat(i, sep_timestamps_history[i].shape[0]))
            colours.extend([sep_timestamps_history[i].shape[0] / COLOUR_SCALE
                            for _ in range(sep_timestamps_history[i].shape[0])])
        timestamps_history_y = np.array(timestamps_history_y)

        sep_timestamps_history = np.hstack(sep_timestamps_history.flat)
        sep_timestamps_history = np.append(batch_timestamps_history, sep_timestamps_history)
        timestamps_history_y = np.hstack(timestamps_history_y.flat)

        plt.scatter(sep_timestamps_history, timestamps_history_y,
                    s=1,
                    c=colours,
                    cmap=get_cmap("viridis"))
        cbar = plt.colorbar()
        cbar.ax.set_ylabel("NFE")
        plt.xlabel("Time")
        plt.ylabel("Batch Example")
        plt.title("epoch{:02d}_itr{:03d}".format(epoch, itr))
        plt.savefig("{}/epoch{:02d}_itr{:03d}_scatter.png".format(dir,  epoch, itr))
        plt.clf()

        # aggregate data for histogram
        all_data.extend(sep_nfe_history)

    plt.hist(all_data)
    plt.xlabel("NFE")
    plt.ylabel("Freq")
    plt.title("epoch{:02d}".format(epoch))
    plt.savefig("{}/epoch{:02d}_hist.png".format(dir, epoch))
    plt.clf()

for epoch in range(20):
    diffs = []
    for itr in range(num_itrs):
        with open("{}/epoch{}_itr{}_batch_nfe_history.pickle".format(dir, epoch, itr), "rb") as f:
            batch_nfe_history = pickle.load(f)

        with open("{}/epoch{}_itr{}_sep_nfe_history.pickle".format(dir, epoch, itr), "rb") as f:
            sep_nfe_history = pickle.load(f)
        # compare batch NFEs to mean for all batch examples
        diff = batch_nfe_history - np.mean(sep_nfe_history)
        diffs.append(diff)

    print("Epoch: {:02d} Mean Diff: {:02.2f}".format(epoch, np.mean(diffs)))
