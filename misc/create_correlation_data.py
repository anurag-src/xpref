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

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_path", None, "Path to model checkpoint.")
flags.DEFINE_boolean("withheld_goals", False, "Whether to use the separate set of withheld goal states")
flags.DEFINE_boolean(
    "restore_checkpoint", True,
    "Restore model checkpoint. Disabling loading a checkpoint is useful if you "
    "want to measure performance at random initialization.")
flags.DEFINE_string("device", "cuda:0", "The compute device.")

ModelType = SelfSupervisedModel

def setup():
    """Load the latest embedder checkpoint and dataloaders."""
    config = utils.load_config_from_dir(FLAGS.experiment_path)
    model = common.get_model(config)

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

    for class_name, class_loader in downstream_loader.items():
        logging.info("Embedding %s.", class_name)
        for batch in tqdm(iter(class_loader), leave=False):
            out = model.infer(batch["frames"].to(device))