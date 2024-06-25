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

"""Teleop the agent and visualize the learned reward."""
import numpy as np
from absl import app
from absl import flags
from absl import logging
from configs.constants import EMBODIMENTS
from configs.constants import XMAGICAL_EMBODIMENT_TO_ENV_NAME
from ml_collections import config_flags
import utils
from xmagical.utils import KeyboardEnvInteractor
import matplotlib.pyplot as plt
import json
import os
import torch
from tqdm import tqdm
from xirl import common
import time

FLAGS = flags.FLAGS

flags.DEFINE_enum("embodiment", "mediumstick", EMBODIMENTS,
                  "The agent embodiment.")

config_flags.DEFINE_config_file(
    "reward_config",
    "base_configs/pretrain.py",
    "File path to the training hyperparameter configuration.",
)

config_flags.DEFINE_config_file(
    "rl_config",
    "base_configs/rl.py",
    "File path to the training hyperparameter configuration.",
)

def main(_):
    env_name = XMAGICAL_EMBODIMENT_TO_ENV_NAME[FLAGS.embodiment]
    env = utils.make_env(env_name, seed=3)  # This env is a proxy, we won't actually be taking actions through it but we use it to access the learned reward
    reward_config = FLAGS.reward_config
    rl_config = FLAGS.rl_config
    device = torch.device("cuda:0") # To prevent this script from crashing any running experiments

    torch.random.manual_seed(0)

    # Load factories.
    (
        model,
        optimizer,
        pretrain_loaders,
        downstream_loaders,
        trainer,
        eval_manager,
    ) = common.get_factories(reward_config, device)

    if rl_config.reward_wrapper.type is not None:
        env = utils.wrap_learned_reward(env, rl_config)

    loader = downstream_loaders
    traj_names = []
    t = str(int(time.time()))
    for class_name, class_loader in downstream_loaders["valid"].items():
        logging.info("Embedding %s.", class_name)
        file_name = f"{t}-{class_name}.csv"

        traj_rewards = []
        for batch in tqdm(iter(class_loader), leave=False):
            path = batch["video_name"][0]
            print(path)
            traj_names.append(path)

            reward_list = json.load(open(os.path.join(path, "rewards.json")))
            gt_reward = sum(reward_list)

            frames = batch['frames'].to(device).squeeze(0)
            eps_reward = []
            for f in frames:
                # You need to permute the image to be (W, H, C) for the peak_reward method to work
                r = env.peak_reward(f.unsqueeze(0).unsqueeze(0).to(device))
                eps_reward.append(r)
            eps_r = sum(eps_reward)
            traj_rewards.append([gt_reward, eps_r])
            print(f"GT reward: {gt_reward}, Learned_r: {eps_r}")

        rew_np = np.array(traj_rewards)
        np.savetxt(file_name, rew_np, delimiter=",")
    json.dump(traj_names, open(f'{t}.json', 'w'))


if __name__ == "__main__":
    app.run(main)
