import numpy as np
from utils import load_config_from_dir
import os
import torch
import yaml

from xprefs.pref_loader import PreferenceLoader
from xprefs.trajectory_loader import TrajectoryLoader
from xprefs.reward_predictor import XPrefsRewardTrainer


def calculate_normalization_denominator(exp_dir):
    """
    Extract the training data from the config of the experiment. Sample N trajectories, set kappa equal to the one that the furthest from Goal
    TODO: Extend this to work for RLHF framework too.
    """
    MAX_TRAJECTORIES_PER_EMBODIMENT = 100
    MEMORY_LIMIT = 50000

    try:
        config = load_config_from_dir(exp_dir)
    except Exception as e:
        raise Exception(f"Config could not be loaded! {e}")

    try:
        goal_file = os.path.join(exp_dir, "goal_embedding.csv")
        goal = torch.Tensor(np.loadtxt(goal_file, delimiter=','))
    except Exception as e:
        raise Exception(f"Attempted to Load Goal file but encountered error! {e}")

    # Load training dataset
    training_trajectories = TrajectoryLoader.full_dataset_from_config(config, train=True)
    training_preferences = PreferenceLoader(config, train=True).preferences
    trainer = XPrefsRewardTrainer(config, exp_dir)

    # Maximize distance of starting frame to goal, set kappa = max
    maxes_across_embodiments = []
    for embodiment in config.data.train_embodiments:

        # Pull trajectories associated with the preferences here.
        trajectory_set = training_preferences.loc[training_preferences["o1_embod"] == embodiment][:MEMORY_LIMIT]
        if len(trajectory_set) == 0:
            continue
        trajectory_ids = list(set(trajectory_set["o1_id"]))[:MAX_TRAJECTORIES_PER_EMBODIMENT]

        tensors = []
        for i in range(0, MAX_TRAJECTORIES_PER_EMBODIMENT):
            tensors.append(training_trajectories.get_item(embodiment, trajectory_ids[i])["frames"][0])
        stacked_t = torch.stack(tensors)
        stacked_t = stacked_t.unsqueeze(1).to(trainer.device)

        with torch.no_grad():
            embeddings = trainer.model.forward(stacked_t).embs
            g_e = torch.squeeze(goal)
            g_e = g_e.repeat(len(embeddings), 1).to(trainer.device)
            goal_diff = g_e - embeddings
            dist_to_reward_o = torch.norm(goal_diff, dim=1)
            maxes_across_embodiments.append(torch.max(dist_to_reward_o).cpu().item())

    print(maxes_across_embodiments)
    kappa = max(maxes_across_embodiments)

    # Save kappa to the experiment directory
    with open(os.path.join(exp_dir, "normalize_embedding.yaml"), "w") as fp:
        yaml.dump({"kappa": kappa, "trajectories_per_emb": MAX_TRAJECTORIES_PER_EMBODIMENT}, fp)

"""
This file can be run independently, but was intended to be called externally at the conclusion of learning an embedding model
Connor Mattson
"""

if __name__ == "__main__":
    calculate_normalization_denominator("/home/connor/Documents/Xpref/experiments/traj_num_cross_20k")