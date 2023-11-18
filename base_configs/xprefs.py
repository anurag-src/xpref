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

"""Default SAC config values."""

import ml_collections
import os

def get_config():
    """Returns default config."""
    config = ml_collections.ConfigDict()

    """
    Specify Training Data
    """
    config.data = ml_collections.ConfigDict()
    config.data.demonstrations_root = os.path.expanduser("~/Documents/Xpref/trajectories_num_blocks/")
    config.data.preference_type = "cross_embodiment"  # Can be one of ["cross_embodiment", "same_embodiment", "combined"]
    config.data.truncate_training_preferences = 1000
    config.data.truncate_validation_loop = 1000
    config.data.truncate_testing_preferences = 1000
    config.data.goal_examples = os.path.expanduser("~/Documents/Xpref/goal_examples")
    config.data.truncate_goals = 200
    config.data.train_embodiments = ["longstick","shortstick","gripper"]
    config.data.validation_embodiments = ["longstick","shortstick","gripper"]
    # config.data.train_embodiments = ["longstick", "shortstick", "gripper"]
    # config.data.validation_embodiments = ["longstick", "shortstick", "gripper"]
    config.data.downstream_embodiments = ["mediumstick"]

    """
    Define Information about Experiment Output
    """
    config.experiments = ml_collections.ConfigDict()
    config.experiments.root = os.path.expanduser("~/Documents/Xpref/experiments")

    """
    Define Information About Trajectory Sampling
    """
    config.sampler = ml_collections.ConfigDict()
    config.sampler.stride = 1  # The stride with which to import video data

    """
    Define Data Transformations
    """
    config.data_augmentation = ml_collections.ConfigDict()
    config.data_augmentation.image_size = (112, 112)
    config.data_augmentation.train_transforms = [
        "global_resize",
        # "random_resized_crop",
        # "color_jitter",
        # "grayscale",
        # "gaussian_blur",
        # "normalize",
    ]
    config.data_augmentation.eval_transforms = [
        "global_resize",
        # "normalize",
    ]

    """
    Define IRL Parameters
    """
    config.irl = ml_collections.ConfigDict()
    config.irl.learning_type = "RLHF"  # Can be ["Xprefs", "RLHF"]
    config.irl.recompute_goal_every = None
    config.irl.train_max_iters = 4000
    config.irl.eval_every = 100
    config.irl.batch_size = 10
    config.irl.embedding_size = 32
    config.irl.lr = 1e-3
    config.irl.average_learned_reward = True
    config.irl.checkpointing_frequency = 5_000
    config.irl.early_terminate_after_loss_equals = 1e-4

    return config
