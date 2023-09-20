import matplotlib.pyplot as plt
import torch
import os
from configs.xmagical.pretraining.tcc import get_config
from xirl import factory
from torchkit import CheckpointManager

CHECKPOINTS = {
    "TCC Only": "/home/connor/Documents/Xpref/experiments/09-19-23-TCCOnly",
    "TCC + XPrefs": "/home/connor/Documents/Xpref/experiments/09-19-23-TCCandXprefs",
    "Xprefs Only": "/home/connor/Documents/Xpref/experiments/09-19-23-XPrefsOnly"
}

EMBODIMENT_TARGET = "shortstick"
TRAJECTORY_INDEX = 22
GOALSET_PATH = os.path.expanduser("~/Documents/Xpref/goal_examples")
LIM_GOALS_PER_EMBODIMENT = 10

CONFIG = get_config()
EVAL_EMBODIMENTS = tuple(["gripper", "shortstick", "mediumstick", "longstick"])
CONFIG.data.pretrain_action_class = EVAL_EMBODIMENTS
CONFIG.data.down_stream_action_class = EVAL_EMBODIMENTS

X_MAGICAL_DATA_PATH = os.path.expanduser("~/Documents/Xpref/xmagical_0.5")
CONFIG.data.root = X_MAGICAL_DATA_PATH

def load_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # device = "cpu"
    return device

def load_preference_dataset(split_type="train", debug=False):
    dataset = factory.full_dataset_from_config(CONFIG, False, split_type, debug)
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

def eval_goal_embedding(model, goals):
    with torch.no_grad():
        return calculate_goal_embedding(model, goals, eval=True)

def calculate_goal_embedding(model, goal_dataloader, eval=False, device="cuda"):
    if not eval:
        model.train()
    else:
        model.eval()

    sum = None
    total_embeddings = 0
    for batch in goal_dataloader:
        frames = batch["frames"].to(device)
        out = model(frames).embs
        total_embeddings += len(out)
        if sum is None:
            sum = torch.sum(out, dim=0)
        else:
            sum += torch.sum(out, dim=0)
    # Average the sum of embeddings
    return sum / total_embeddings

def load_model(checkpoint_dir):
    m = factory.model_from_config(CONFIG)
    checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir,
        model=m,
        # optimizer=optimizer,
    )
    checkpoint_manager.restore_or_initialize()
    m.eval()
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
    traj_data = load_preference_dataset("valid", debug=False)
    goal_data = load_goal_frames("train")
    device = load_device()
    observation = traj_data.get_item(EMBODIMENT_TARGET, TRAJECTORY_INDEX)
    fig, axes = plt.subplots(nrows=len(CHECKPOINTS), ncols=1, sharex=True)

    i = 0
    for cp in CHECKPOINTS:
        m = load_model(CHECKPOINTS[cp]).to(device)
        goal_embedding = eval_goal_embedding(m, goal_data)
        rewards = r_from_traj(observation, m, goal_embedding, device=device)
        x = [i for i in range(len(observation["frames"]))]
        y = rewards.cpu().numpy()
        axes[i].plot(x, y, label=cp.lower())
        axes[i].set_title(cp)
        i += 1

    plt.ylabel("Negative Distance to Goal")
    plt.xlabel("Trajectory Index")
    # plt.title("Comparison of Inferred Reward over trajectory")
    plt.show()

