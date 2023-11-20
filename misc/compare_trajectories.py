import matplotlib.pyplot as plt
import torch
import os
from base_configs.xprefs import get_config as get_xprefs_config
from xprefs.trajectory_loader import TrajectoryLoader
from xirl import factory
import numpy as np
from torchkit import CheckpointManager
from xprefs.reward_predictor import XPrefsRewardTrainer

CHECKPOINTS = {
    # "TCC Only (XMagical)": "/home/connor/Documents/Xpref/experiments/09-26-23-TCCXMagical",
    # "TCC Only (MQME 0.5)": "/home/connor/Documents/Xpref/experiments/09-26-23-TCCMQME",
    # "TCC + XPrefs (MQME 0.5)": "/home/connor/Documents/Xpref/experiments/09-26-23-TCCandXPrefs",
    # "Xprefs Only (Dynamic Goal $\phi$)" : "/home/connor/Documents/Xpref/experiments/09-26-23-XPrefsOnlyDynamicGoal",
    "Xprefs Only (Static Goal $\phi$)" : "/home/connor/Documents/Xpref/experiments/traj1_11-13-23"
}

EMBODIMENT_TARGETS = ["mediumstick", "mediumstick"]
GOOD_TRAJECTORY_INDEX = 299
BAD_TRAJECTORY_INDEX = 1800
# OTHER_TRAJECTORY_INDEX = 119

CONFIG = get_xprefs_config()
EVAL_EMBODIMENTS = tuple(["gripper", "shortstick", "mediumstick", "longstick"])
CONFIG.data.pretrain_action_class = EVAL_EMBODIMENTS
CONFIG.data.down_stream_action_class = EVAL_EMBODIMENTS

X_MAGICAL_DATA_PATH = os.path.expanduser("~/Documents/Xpref/trajectories")
CONFIG.data.root = X_MAGICAL_DATA_PATH

def load_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    # device = "cpu"
    return device

def load_preference_dataset():
    dataset = TrajectoryLoader.full_dataset_from_config(CONFIG, True)
    return dataset

def calculate_goal_embedding(exp_dir, device="cuda"):
    goal_file = os.path.join(exp_dir, "goal_embedding.csv")
    goal = np.loadtxt(goal_file, delimiter=',')
    return torch.tensor(goal).to(device)

def load_model(checkpoint_dir):
    m = XPrefsRewardTrainer(CONFIG, checkpoint_dir).model
    return m

def r_from_traj(observation, model, goal_embedding, device):
    with torch.no_grad():
        o_frames = torch.stack([observation["frames"].to(device)])
        embed_o = model(o_frames).embs.squeeze()
        goal_embedding = torch.squeeze(goal_embedding)
        goal_example_o_size = goal_embedding.repeat(len(embed_o), 1)
        goal_diff = goal_example_o_size - embed_o
        dist_to_reward_o = torch.norm(goal_diff, dim=1)
        return -dist_to_reward_o

if __name__ == "__main__":
    traj_data = load_preference_dataset()
    device = load_device()
    print(len(traj_data))
    observation_1 = traj_data.get_item(EMBODIMENT_TARGETS[0], GOOD_TRAJECTORY_INDEX, eval=False)
    observation_2 = traj_data.get_item(EMBODIMENT_TARGETS[1], BAD_TRAJECTORY_INDEX, eval=False)
    # observation_3 = traj_data.get_item(EMBODIMENT_TARGETS[2], OTHER_TRAJECTORY_INDEX, eval=False)
    fig, axes = plt.subplots(nrows=len(CHECKPOINTS), ncols=1, sharex=True)

    i = 0
    for cp in CHECKPOINTS:
        m = load_model(CHECKPOINTS[cp]).to(device)
        goal_embedding = calculate_goal_embedding(CHECKPOINTS[cp], device=device)
        rewards_1 = r_from_traj(observation_1, m, goal_embedding, device=device)
        rewards_2 = r_from_traj(observation_2, m, goal_embedding, device=device)
        # rewards_3 = r_from_traj(observation_3, m, goal_embedding, device=device)

        x_1 = [i for i in range(len(observation_1["frames"]))]
        x_2 = [i for i in range(len(observation_2["frames"]))]
        # x_3 = [i for i in range(len(observation_3["frames"]))]

        y_1 = rewards_1.cpu().numpy()
        y_2 = rewards_2.cpu().numpy()
        # y_3 = rewards_3.cpu().numpy()

        try:
            axes[i].plot(x_1, y_1, label=cp.lower(), c="blue")
            axes[i].plot(x_2, y_2, label=cp.lower(), c="red")
            # axes[i].plot(x_3, y_3, label=cp.lower(), c="green")
            axes[i].set_title(cp)
            axes[i].set_ylim(min(min(y_1), min(y_2)), 0)
        except TypeError:
            axes.plot(x_1, y_1, label=f"{EMBODIMENT_TARGETS[0]}-{GOOD_TRAJECTORY_INDEX}", c="blue")
            axes.plot(x_2, y_2, label=f"{EMBODIMENT_TARGETS[1]}-{BAD_TRAJECTORY_INDEX}", c="red")
            # axes.plot(x_3, y_3, label=f"{EMBODIMENT_TARGETS[2]}-{OTHER_TRAJECTORY_INDEX}", c="green")
            axes.set_title(cp)
            axes.set_ylim(min(min(y_1), min(y_2)), 0)
        i += 1

    plt.ylabel("Negative Distance to Goal")
    plt.xlabel("Trajectory Index")
    plt.legend()
    # plt.title("Comparison of Inferred Reward over trajectory")
    plt.show()

