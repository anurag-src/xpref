# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compute and store the mean goal embedding using a trained model."""

import os
import typing

from absl import app
from absl import flags
from absl import logging
import numpy as np
import torch
from torchkit import CheckpointManager
from tqdm.auto import tqdm
import utils
from xirl import common, factory
from xirl.models import SelfSupervisedModel
from base_configs.xprefs import get_config as get_xprefs_config
from xprefs.calculate_kappa import calculate_normalization_denominator

# pylint: disable=logging-fstring-interpolation

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_path", None, "Path to model checkpoint.")
flags.DEFINE_boolean("withheld_goals", False, "Whether to use the separate set of withheld goal states")
flags.DEFINE_boolean(
    "restore_checkpoint", True,
    "Restore model checkpoint. Disabling loading a checkpoint is useful if you "
    "want to measure performance at random initialization.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")

ModelType = SelfSupervisedModel
DataLoaderType = typing.Dict[str, torch.utils.data.DataLoader]


def embed(
        model,
        downstream_loader,
        device,
):
    """Embed the stored trajectories and compute mean goal embedding."""
    goal_embs = []
    init_embs = []
    for class_name, class_loader in downstream_loader.items():
        logging.info("Embedding %s.", class_name)
        for batch in tqdm(iter(class_loader), leave=False):
            out = model.infer(batch["frames"].to(device))
            emb = out.numpy().embs
            init_embs.append(emb[0, :])
            goal_embs.append(emb[-1, :])
    goal_emb = np.mean(np.stack(goal_embs, axis=0), axis=0, keepdims=True)
    dist_to_goal = np.linalg.norm(
        np.stack(init_embs, axis=0) - goal_emb, axis=-1).mean()
    distance_scale = 1.0 / dist_to_goal
    return goal_emb, distance_scale

def distance_calc(
        model,
        downstream_loader,
        goal_emb,
        device,
):
    """Embed the stored trajectories and compute mean goal embedding."""
    init_embs = []
    for class_name, class_loader in downstream_loader.items():
        logging.info("Embedding %s.", class_name)
        for batch in tqdm(iter(class_loader), leave=False):
            out = model.infer(batch["frames"].to(device))
            emb = out.numpy().embs
            init_embs.append(emb[0, :])
    dist_to_goal = np.linalg.norm(
        np.stack(init_embs, axis=0) - goal_emb, axis=-1).mean()
    distance_scale = 1.0 / dist_to_goal
    return goal_emb, distance_scale

def embed_withheld_goals(
        model,
        downstream_loader,
        device,
):
    """Embed the stored trajectories and compute mean goal embedding."""
    goal_embs = []
    init_embs = []
    for class_name, class_loader in downstream_loader.items():
        logging.info("Embedding %s.", class_name)
        for batch in tqdm(iter(class_loader), leave=False):
            out = model.infer(batch["frames"].to(device))
            emb = out.embs.numpy()
            for i in range(len(emb)):
                goal_embs.append(emb[i, :])
    goal_emb = np.mean(np.stack(goal_embs, axis=0), axis=0)
    print(goal_emb.shape)
    return goal_emb, 1.0


def load_withheld_goal_frames(config, split_type="Train", debug=False):
    dataset = factory.goal_dataset_from_config(config, config.data.goal_examples,
                                               config.data.truncate_goals, False, split_type,
                                               debug)
    return torch.utils.data.DataLoader(
        dataset,
        # collate_fn=dataset.collate_fn,
        batch_size=50,
        num_workers=4 if torch.cuda.is_available() and not debug else 0,
        pin_memory=torch.cuda.is_available() and not debug,
    )


def setup():
    """Load the latest embedder checkpoint and dataloaders."""
    config = utils.load_config_from_dir(FLAGS.experiment_path)
    model = common.get_model(config)

    if FLAGS.withheld_goals:
        xprefs_config = get_xprefs_config()
        mixed_goal_loader = load_withheld_goal_frames(xprefs_config, "train")
        downstream_loaders = {"mixed_goals": mixed_goal_loader}
    else:
        downstream_loaders = common.get_downstream_dataloaders(config, False)["train"]
    checkpoint_dir = os.path.join(FLAGS.experiment_path, "checkpoints")
    if FLAGS.restore_checkpoint:
        checkpoint_manager = CheckpointManager(checkpoint_dir, model=model)
        global_step = checkpoint_manager.restore_or_initialize()
        logging.info("Restored model from checkpoint %d.", global_step)
    else:
        logging.info("Skipping checkpoint restore.")
    return model, downstream_loaders


def main(_):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, downstream_loader = setup()
    model.to(device).eval()

    if FLAGS.withheld_goals:
        # Distance Scale is 1.0 by default in this case
        goal_emb, _ = embed_withheld_goals(model, downstream_loader, device)

        # Revert to default to get the distance metric
        FLAGS.withheld_goals = False
        _, downstream_loader = setup()
        _, distance_scale = distance_calc(model, downstream_loader, goal_emb, device)
    else:
        goal_emb, distance_scale = embed(model, downstream_loader, device)

    utils.save_pickle(FLAGS.experiment_path, distance_scale, "distance_scale.pkl")
    utils.save_pickle(FLAGS.experiment_path, goal_emb, "goal_emb.pkl")

if __name__ == "__main__":
    flags.mark_flag_as_required("experiment_path")
    app.run(main)
