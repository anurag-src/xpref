import matplotlib.pyplot as plt
import numpy as np

RESULTS = "/home/connor/Documents/Xpref/pretrain_runs/env_name=SweepToTop-Mediumstick-State-Allo-TestLayout-v0_reward=learned_reward_type=reward_prediction_from_prefs_mode=same_algo=xprefs_uid=d93618d2-1ae4-4d05-9d52-88b25fa91956/0/reward_tracking.csv"
if __name__ == "__main__":
    data = np.loadtxt(RESULTS, delimiter=",")
    gt = data[:, 0]
    learned = data[:, 1]
    x = data[:, 2]

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('RL Training Steps (SAC)')
    ax1.set_ylabel('Avg. Ground Truth Reward', color=color)
    ax1.plot(x, gt, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Avg. Inferred Reward', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, learned, color=color)
    ax2.tick_params(axis='y', labelcolor=color)


    plt.title("SAC Reward Progress during Training on Inferred Reward (Mediumstick)")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()