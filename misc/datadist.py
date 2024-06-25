import os
import numpy as np

from xirl import factory
from xprefs.pref_loader import PreferenceLoader
from xprefs.trajectory_loader import TrajectoryLoader
from xprefs.reward_predictor import XPrefsRewardTrainer
from base_configs.xprefs import get_config as get_xprefs_config
from configs.xmagical.pretraining.tcc import get_config

import matplotlib.pyplot as plt

def get_num_blocks_pushed_in(root, embodiment, i):
    path_to_dir = os.path.join(root, "train", embodiment, str(i), "rewards.json")
    if not os.path.exists(path_to_dir):
        raise Exception("Folder not found: ", path_to_dir)
    with open(path_to_dir, "r") as f:
        rewards = eval(f.readlines()[0])
    return max(rewards)


def evaluate_data_for_n_preferences(n):
    config = get_xprefs_config()
    config.data.train_embodiments = ["longstick","shortstick","mediumstick","gripper"]
    config.data.validation_embodiments = ["longstick", "shortstick", "mediumstick", "gripper"]
    config.data.preference_type = "cross_embodiment"
    config.data.truncate_training_preferences = n

    training_trajectories = TrajectoryLoader.full_dataset_from_config(config, train=True)
    training_preferences = PreferenceLoader(config, train=True).preferences

    blocks_tree = {}
    for e in config.data.train_embodiments:
        blocks_tree[e] = {}
        ids_o1 = list(training_preferences.loc[training_preferences["o1_embod"] == e]["o1_id"])
        ids_o2 = list(training_preferences.loc[training_preferences["o2_embod"] == e]["o2_id"])
        for i in set(ids_o1 + ids_o2):
            blocks_tree[e][i] = get_num_blocks_pushed_in(config.data.demonstrations_root, e, i)

    # Y-AXIS: O2
    # X-AXIS: O1
    answers = np.zeros((4, 4))

    for index, row in training_preferences.iterrows():
        x_index = int(float(blocks_tree[row["o1_embod"]][int(row["o1_id"])]) * 3)
        y_index = int(float(blocks_tree[row["o2_embod"]][int(row["o2_id"])]) * 3)
        answers[y_index][x_index] += 1

    answers /= np.sum(answers)
    print(answers[::-1])
    answers = np.transpose(answers[::-1])


    fig, ax = plt.subplots()
    im = ax.imshow(answers)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(answers[0])), labels=[0, 1, 2, 3])
    ax.set_yticks(np.arange(len(answers)), labels=[3, 2, 1, 0])

    # Loop over data dimensions and create text annotations.
    for i in range(len(answers)):
        for j in range(len(answers[i])):
            text = ax.text(i, j, answers[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Frequency of blocks pushed in by preference pairs ($\phi_1$ > $\phi_2$)")
    ax.set_xlabel("Blocks pushed in by $\phi_1$")
    ax.set_ylabel("Blocks pushed in by $\phi_2$")
    fig.tight_layout()
    plt.show()

def evaluate_bucket_distribution():
    dataset = factory.dataset_from_config(get_config(), False, "train", False, with_reward=True)
    embodiments = {}
    for r, e_class, ind in dataset.reward_set:
        if not e_class in embodiments:
            embodiments[e_class] = []
        embodiments[e_class].append(r)

    fig, axes = plt.subplots(len(embodiments), 1, sharex=True, sharey=True)
    colors = ["red", "blue", "green", "purple"]
    i = 0
    for k in embodiments:
        axes[i].hist(embodiments[k], bins=20, alpha=0.5, color=colors[i], label=k[k.rfind("/"):])
        # axes[i].set_xlabel("Average Reward")
        axes[i].set_xlabel("Cumulative Reward")
        # axes[i].set_xlabel("No. Blocks Pushed In")
        axes[i].set_ylabel("Frequency")
        i += 1
    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels)

    fig.suptitle("Demonstration Avg. Reward Distribution per Embodiment Type")
    # plt.legend()
    plt.show()



if __name__ == "__main__":
    # evaluate_data_for_n_preferences(20000)
    evaluate_bucket_distribution()