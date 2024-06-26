import matplotlib.pyplot as plt
import os
import numpy as np

methods = [
    "Ground Truth",
    "XIRL",
    "XIRL Mixed",
    "Goal Classifier",
    "X-RLHF",
    "XTriplets",
    "XIRL-Buckets",
    "XPrefs",
    "XPrefs_0G",
]

def load_mean_and_std_error(method):
    csvs = []
    sample_count = len(os.listdir(os.path.join('../data/Final_Comparison_Graph_Data', method)))
    for folder in os.listdir(os.path.join('../data/Final_Comparison_Graph_Data', method)):
        csvs.append(np.genfromtxt(os.path.join('../data/Final_Comparison_Graph_Data', method, folder, "reward_tracking.csv"), delimiter=",", dtype=np.float32))
    combined = np.stack(csvs)
    mean = np.mean(combined, axis=0)
    std_err = np.std(combined, axis=0) / np.sqrt(sample_count)
    return mean, std_err

if __name__ == "__main__":
    fig, ax = plt.subplots(1)
    for m in methods:
        u, err = load_mean_and_std_error(m)
        x = u[:, -1]
        y = u[:, 0]
        err = err[:, 0]
        ax.plot(x, y, label=m if m != "XPrefs_0G" else "XPrefs (g=0)")
        ax.fill_between(x, y + err, y - err, alpha=0.15)

    plt.ylabel("Cumulative Ground Truth Reward")
    plt.xlabel("SAC Training Steps")
    plt.title("RL Performance of Learned Rewards Evaluated Under the GT Reward")
    fig.tight_layout()
    plt.legend()
    plt.savefig("results-comparision.pdf", format="pdf", bbox_inches="tight")
    # plt.show()
    fig.show()