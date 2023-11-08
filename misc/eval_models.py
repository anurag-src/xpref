import gym
import xmagical
from gym.wrappers import Monitor
import torch
import os
from PIL import Image
xmagical.register_envs()
import collections
import os.path as osp
from typing import Dict

from absl import app
from absl import flags
from absl import logging
from base_configs import validate_config
import gym
from ml_collections import config_dict
from ml_collections import config_flags
import numpy as np
from sac import agent
import torch
from torchkit import CheckpointManager
from torchkit import experiment
from torchkit import Logger
from tqdm.auto import tqdm
import utils

def evaluate(
        policy,
        env,
        num_episodes,
):
    """Evaluate the policy and dump rollout videos to disk."""
    policy.eval()
    stats = collections.defaultdict(list)
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = policy.act(observation, sample=False)
            observation, _, done, info = env.step(action)
        for k, v in info["episode"].items():
            stats[k].append(v)
        if "eval_score" in info:
            stats["eval_score"].append(info["eval_score"])
    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats

model_dir = "~/Documents/Research/Xpref/rl/run_0/0/checkpoints/249999.cpkt"

envname = 'SweepToTop-Mediumstick-Pixels-Allo-Layout-v0'

checkpoint = torch.load(model_dir)



env = gym.make(envname)

