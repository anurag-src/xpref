import matplotlib.pyplot as plt
import numpy as np

RESULTS = "/home/connor/Documents/Xpref/pretrain_runs/env_name=SweepToTop-Mediumstick-State-Allo-TestLayout-v0_reward=learned_reward_type=reward_prediction_from_prefs_mode=cross_algo=xirl_uid=f572e1c7-16b7-47c8-98c1-dbc95b5dedc1/0/reward_tracking.csv"
if __name__ == "__main__":
    data = np.loadtxt(RESULTS, delimiter=",")
    gt = data[:, 0]
    learned = data[:, 1]
    x = data[:, 2]

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('RL Training Steps (SAC)')
    ax1.set_ylabel('Avg. Cumulative Ground Truth Reward', color=color)
    ax1.plot(x, gt, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Avg. Cumulative Learned Reward', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, learned, color=color)
    ax2.tick_params(axis='y', labelcolor=color)


    plt.title("SAC Reward Progress during Training on Inferred Reward (Mediumstick)")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()