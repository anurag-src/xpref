import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

methods = [
    ("gt_reward.txt", "Ground Truth"),
    ("xirl_reward.txt", "XIRL"),
    ("xirl_mqme_reward.txt", "XIRL Mixed"),
    ("goal_classifier.txt", "Goal Classifier"),
    ("rlhf_reward.txt", "X-RLHF"),
    ("triplets.txt", "XTriplets"),
    ("buckets_reward.txt", "XIRL-Buckets"),

    ("xprefs_reward.txt", "XPrefs"),
    # ("xprefs_0G.txt", "XPrefs (g=0)"),


    # ("4_buckets.txt", "Buckets-4"),
]

def load_data(method):
    arr = np.loadtxt(os.path.join('../data/bad_trajectory_rewards', method))
    return arr

if __name__ == "__main__":
    fig, ax = plt.subplots(7, 1, figsize=(8, 9), sharex=True)
    for i in range(len(methods)):
        m = methods[i][0]
        n = methods[i][1]
        arr = load_data(m)
        # print(df)
        x = range(len(arr))
        y = arr
        ax[i].plot(x, y, label=n)
        ax[i].set_title(n)
        ax[i].set_ylabel("Reward")

        if i == 0:
            ax[i].set_ylim(bottom=-0.05, top=1.1)
        if 1 <= i <= 2:
            ax[i].set_ylim(bottom=-1, top=0.05)
        if i == 3:
            ax[i].set_ylim(bottom=-0.05, top=1.1)
        if i == 4:
            ax[i].set_ylim(bottom=-35, top=25)
        if i == 5:
            pass
            # ax[i].set_ylim(bottom=-1.3, top=-0.5)
            # ax[i].set_ylim(bottom=-0.25, top=-0.12)
            # ax[i].set_ylim(bottom=-105, top=-60) # XPrefs
        if i == 6:
            ax[i].set_ylim(bottom=-1, top=0.05)


        # if i == 8:
        #     ax[i].set_ylim(bottom=-10, top=-4)
        # ax.fill_between(x, y + err, y - err, alpha=0.15)

    # plt.ylabel("Training Loss")
    plt.xlabel("Time (timesteps)")
    # plt.title("")
    fig.tight_layout()
    # plt.legend()
    plt.savefig("dynamic-comparison.pdf", format="pdf", bbox_inches="tight")
    plt.show()