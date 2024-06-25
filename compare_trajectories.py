import matplotlib.pyplot as plt
import torch
import os
from base_configs.xprefs import get_config as get_xprefs_config
from xirl import factory
from torchkit import CheckpointManager
import numpy as np
from xprefs.reward_predictor import XPrefsRewardTrainer
from xprefs.trajectory_loader import TrajectoryLoader

CHECKPOINTS = {
    # "TCC Only (XMagical)": "/home/connor/Documents/Xpref/pretrain_runs/dataset=xmagical_mode=cross_algo=xirl_embodiment=mediumstick/",
    # "TCC Only (MQME 0.5)": "/home/connor/Documents/Xpref/experiments/09-26-23-TCCMQME",
    # "TCC + XPrefs (MQME 0.5)": "/home/connor/Documents/Xpref/experiments/09-26-23-TCCandXPrefs",
    # "Xprefs Only (Test A)" : "/home/connor/Documents/Xpref/experiments/09-19-23-TCCOnly",
    "Xprefs Only (Test B)" : "/home/connor/Documents/Xpref/experiments/traj_num_cross_20k"
}

EMBODIMENT_TARGETS = ["mediumstick", "mediumstick", "mediumstick"]
GOOD_TRAJECTORY_INDEX = 2
BAD_TRAJECTORY_INDEX = 673
OTHER_TRAJECTORY_INDEX = 1995
GOALSET_PATH = os.path.expanduser("~/Documents/Xpref/goal_examples")
LIM_GOALS_PER_EMBODIMENT = 10

CONFIG = get_xprefs_config()
EVAL_EMBODIMENTS = tuple(["gripper", "shortstick", "mediumstick", "longstick"])
CONFIG.data.train_embodiments = EVAL_EMBODIMENTS
CONFIG.data.validation_embodiments = EVAL_EMBODIMENTS

def load_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    # device = "cpu"
    return device

def load_preference_dataset(debug=False):
    dataset = TrajectoryLoader.full_dataset_from_config(CONFIG, False, debug)
    return dataset

def load_goal_frames(split_type="Train", debug=False):
    """
    n: The number of frames to inject into the dataset, if None, use all frames
    """
    dataset = factory.goal_dataset_from_config(CONFIG, GOALSET_PATH, LIM_GOALS_PER_EMBODIMENT, False, split_type, debug)
    return torch.utils.data.DataLoader(
        dataset,
        # collate_fn=dataset.collate_fn,
        batch_size=50,
        num_workers=4 if torch.cuda.is_available() and not debug else 0,
        pin_memory=torch.cuda.is_available() and not debug,
    )

def calculate_goal_embedding(exp_dir, device="cuda"):
    goal_file = os.path.join(exp_dir, "goal_embedding.csv")
    goal = np.loadtxt(goal_file, delimiter=',')
    return torch.tensor(goal).to(device)

def load_model(checkpoint_dir):
    m = XPrefsRewardTrainer(CONFIG, checkpoint_dir)
    m.load_checkpoint_manager()
    m = m.model
    m.eval()
    return m

def r_from_traj(observation, model, goal_embedding, device):
    model.eval()
    with torch.no_grad():
        o_frames = torch.stack([observation["frames"].to(device)])
        embed_o = model(o_frames).embs.squeeze()
        goal_embedding = torch.squeeze(goal_embedding)
        goal_example_o_size = goal_embedding.repeat(len(embed_o), 1)
        goal_diff = goal_example_o_size - embed_o
        dist_to_reward_o = torch.norm(goal_diff, dim=1)
        return -dist_to_reward_o


if __name__ == "__main__":
    traj_data = load_preference_dataset(debug=False)
    goal_data = load_goal_frames("train")
    device = load_device()
    print(len(traj_data))
    observation_1 = traj_data.get_item(EMBODIMENT_TARGETS[0], GOOD_TRAJECTORY_INDEX, eval=False)
    observation_2 = traj_data.get_item(EMBODIMENT_TARGETS[1], BAD_TRAJECTORY_INDEX, eval=False)
    observation_3 = traj_data.get_item(EMBODIMENT_TARGETS[2], OTHER_TRAJECTORY_INDEX, eval=False)
    fig, axes = plt.subplots(nrows=len(CHECKPOINTS), ncols=1, sharex=True)

    i = 0
    for cp in CHECKPOINTS:
        m = load_model(CHECKPOINTS[cp]).to(device)
        goal_embedding = calculate_goal_embedding(CHECKPOINTS[cp], device=device)
        rewards_1 = r_from_traj(observation_1, m, goal_embedding, device=device)
        rewards_2 = r_from_traj(observation_2, m, goal_embedding, device=device)
        rewards_3 = r_from_traj(observation_3, m, goal_embedding, device=device)

        x_1 = [i for i in range(len(observation_1["frames"]))]
        x_2 = [i for i in range(len(observation_2["frames"]))]
        x_3 = [i for i in range(len(observation_3["frames"]))]

        y_1 = rewards_1.cpu().numpy()
        y_2 = rewards_2.cpu().numpy()
        y_3 = rewards_3.cpu().numpy()

        print(f"Experiment {cp}:")
        print(f"Average {EMBODIMENT_TARGETS[0]}-{GOOD_TRAJECTORY_INDEX}: {np.mean(y_1)}")
        print(f"Average {EMBODIMENT_TARGETS[1]}-{BAD_TRAJECTORY_INDEX}: {np.mean(y_2)}")
        print(f"Average {EMBODIMENT_TARGETS[2]}-{OTHER_TRAJECTORY_INDEX}: {np.mean(y_3)}")

        try:
            axes[i].plot(x_1, y_1, label=f"{EMBODIMENT_TARGETS[0]}-{GOOD_TRAJECTORY_INDEX}", c="blue")
            axes[i].plot(x_2, y_2, label=f"{EMBODIMENT_TARGETS[1]}-{BAD_TRAJECTORY_INDEX}", c="red")
            axes[i].plot(x_3, y_3, label=f"{EMBODIMENT_TARGETS[2]}-{OTHER_TRAJECTORY_INDEX}", c="green")
            axes[i].set_title(cp)
            axes[i].set_ylim(min(min(y_1), min(y_2), min(y_3)), 0)
        except TypeError:
            axes.plot(x_1, y_1, label=f"{EMBODIMENT_TARGETS[0]}-{GOOD_TRAJECTORY_INDEX}", c="blue")
            axes.plot(x_2, y_2, label=f"{EMBODIMENT_TARGETS[1]}-{BAD_TRAJECTORY_INDEX}", c="red")
            axes.plot(x_3, y_3, label=f"{EMBODIMENT_TARGETS[2]}-{OTHER_TRAJECTORY_INDEX}", c="green")
            axes.set_title(cp)
            axes.set_ylim(min(min(y_1), min(y_2), min(y_3)), 0)
        i += 1

    plt.ylabel("Negative Distance to Goal")
    plt.xlabel("Trajectory Index")
    plt.legend()
    # plt.title("Comparison of Inferred Reward over trajectory")
    plt.show()

