"""
Parse the pickle files
"""
import pickle

dir = "img_results_2019-07-27_18-50_cifar10_experiments"

with open("{}/batch_nfe_history.pickle".format(dir), "rb") as f:
    batch_nfe_history = pickle.load(f)

with open("{}/batch_timestamps_history.pickle".format(dir), "rb") as f:
    batch_timestamps_history = pickle.load(f)

with open("{}/sep_nfe_history.pickle".format(dir), "rb") as f:
    sep_nfe_history = pickle.load(f)

with open("{}/sep_timestamps_history.pickle".format(dir), "rb") as f:
    sep_timestamps_history = pickle.load(f)

print("hi")
