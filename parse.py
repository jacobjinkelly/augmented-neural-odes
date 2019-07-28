"""
Parse the pickle files
"""
import pickle
import torch

import numpy as np
import matplotlib.pyplot as plt

dir = "img_results_2019-07-27_22-39_cifar10_experiments"
num_epochs = 1
num_itrs = 30
batch_size = 256

for epoch in range(num_epochs):
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

        # scaling for scatter plot
        SCALE = 1
        sep_timestamps_history = np.array(sep_timestamps_history)
        sep_timestamps_history_y = []
        for i in range(len(sep_timestamps_history)):
            sep_timestamps_history[i] = np.array(sep_timestamps_history[i])
            sep_timestamps_history_y.append(np.repeat(i, sep_timestamps_history[i].shape[0]))
        sep_timestamps_history_y = np.array(sep_timestamps_history_y)

        sep_timestamps_history = np.hstack(sep_timestamps_history.flat)
        sep_timestamps_history_y = np.hstack(sep_timestamps_history_y.flat)

        plt.scatter(sep_timestamps_history, sep_timestamps_history_y)
        plt.savefig("{}/epoch{:02d}_itr{:03d}_scatter.png".format(dir,  epoch, itr))
        plt.clf()

        all_data.extend(sep_nfe_history)

    plt.hist(all_data)
    plt.savefig("{}/epoch{:02d}_hist.png".format(dir, epoch))
    plt.clf()
