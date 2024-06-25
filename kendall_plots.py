import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLASSES = [
    "gripper",
    "longstick",
    "shortstick",
    "mediumstick",
]

TREATMENTS = {
    "xirl": "XIRL",
    "xirl-mqme": "XIRL Mixed",
    "goal_classifier": "Goal Classifier",
    "rlhf": "X-RLHF",
    "triplets": "XTriplets",
    "buckets": "XIRL-Buckets",
    "xprefs" : "XPrefs"
}

# TREATMENTS = {
#     "buckets": "32 Buckets",
#     "buckets4": "4 Buckets",
# }

ROOT = "data/correl"


def get_data(cl, treatment):
    path = os.path.join(ROOT, treatment, f"{cl}.csv")
    data = np.genfromtxt(path, delimiter=",")
    return data


def normalize_y(data):
    min_x, max_x = np.min(data[:, 0]), np.max(data[:, 0])
    min_y, max_y = np.min(data[:, 1]), np.max(data[:, 1])
    data[:, 1] = (data[:, 1] - min_y) / (max_y - min_y)
    data[:, 1] = (data[:, 1] * (max_x - min_x)) + min_x
    return data

def correlation_plot():
    fig = plt.figure(constrained_layout=False)
    fig.suptitle('Learned and GT Reward Correlation For XMagical Demonstrations')

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=len(TREATMENTS), ncols=1)

    for i, treatment in enumerate(TREATMENTS):

        treatment_title = TREATMENTS[treatment]
        subfigs[i].suptitle(f'{treatment_title}', fontsize=11)

        axs = subfigs[i].subplots(nrows=1, ncols=len(CLASSES))

        for col, ax in enumerate(axs):
            cl = CLASSES[col]
            data = get_data(cl, treatment)
            data = normalize_y(data)
            x, y = data[:, 0], data[:, 1]
            ax.scatter(x, y, s=5)
            if col == 0:
                ax.set_ylabel(r"Cumulative $\hat{r}$")
            if i == len(TREATMENTS) - 1:
                ax.set_xlabel("Cumulative GT $r$")
            ax.set_title(cl, fontsize=9)

    # plt.subplots_adjust(top=0.8)
    plt.show()

def kendall_calculation():

    for t, treatment in enumerate(TREATMENTS):
        for c, cl in enumerate(CLASSES):
            data = get_data(cl, treatment)
            data = normalize_y(data)
            X = data
            concordant_count = 0
            discordant_count = 0
            ties = 0
            estim_total = (len(X) * (len(X) - 1)) / 2
            total = 0
            for i in range(len(X)):
                for j in range(i + 1, len(X)):
                    # Get pair of points
                    x1, y1 = X[i][0], X[i][1]
                    x2, y2 = X[j][0], X[j][1]
                    if x1 > x2 and y1 > y2:
                        concordant_count += 1
                    elif x1 < x2 and y1 < y2:
                        concordant_count += 1
                    elif abs(x1 - x2) < 10e-6 or abs(y1 - y2) < 10e-6:
                        ties += 1
                    else:
                        discordant_count += 1
                    total += 1

            tau = (concordant_count - discordant_count) / total
            accuracy = concordant_count / estim_total
            print(f"Kendall's Tau for {treatment}, Total Ties: {ties}, {cl}: {tau:.2f}, Accuracy: {accuracy:.2f}")

    # plt.subplots_adjust(top=0.8)
    plt.show()

if __name__ == "__main__":
    # kendall_calculation()
    correlation_plot()
