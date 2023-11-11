"""
In this file, we consider the XPrefs method where an average goal embedding is computed and used as the basis for a reward predictor
The model follows the same architecture as TCC.
"""
import os
import time
import torch
import yaml
from xprefs.pref_loader import PreferenceLoader
from configs.xmagical.pretraining.tcc import get_config
from base_configs.xprefs import get_config as get_xprefs_config
from xprefs.trajectory_loader import TrajectoryLoader
from xprefs.reward_predictor import XPrefsRewardTrainer
from xirl import factory
from torchkit import CheckpointManager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_collections import config_dict
import random
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
ConfigDict = config_dict.ConfigDict
XPREFS_CONFIG = get_xprefs_config()

def load_goal_frames(split_type="Train", debug=False):
    dataset = factory.goal_dataset_from_config(XPREFS_CONFIG, XPREFS_CONFIG.data.goal_examples, XPREFS_CONFIG.data.truncate_goals, False, split_type,
                                               debug)
    return torch.utils.data.DataLoader(
        dataset,
        # collate_fn=dataset.collate_fn,
        batch_size=50,
        num_workers=4 if torch.cuda.is_available() and not debug else 0,
        pin_memory=torch.cuda.is_available() and not debug,
    )


def create_experiment_dir(name=None):
    if not os.path.exists(XPREFS_CONFIG.experiments.root):
        os.makedirs(XPREFS_CONFIG.experiments.root)
    if name is None:
        name = str(int(time.time()))
    exp_dir = os.path.join(XPREFS_CONFIG.experiments.root, name)
    os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
        yaml.dump(ConfigDict.to_dict(XPREFS_CONFIG), fp)
    return exp_dir


def train_xprefs():
    # Create an experiment for the trial
    exp_dir = create_experiment_dir()

    # Load the preference data from the config using a Dataloader, returns pandas Dataframe
    training_preferences = PreferenceLoader(XPREFS_CONFIG, train=True).preferences
    validation_preferences = PreferenceLoader(XPREFS_CONFIG, train=False).preferences
    training_trajectories = TrajectoryLoader.full_dataset_from_config(XPREFS_CONFIG, train=True)
    validation_trajectories = TrajectoryLoader.full_dataset_from_config(XPREFS_CONFIG, train=False)

    # Load the trainer
    trainer = XPrefsRewardTrainer(XPREFS_CONFIG, exp_dir)
    trainer.attach_data_to_trainer(training_trajectories, validation_trajectories)
    trainer.attach_prefs_to_trainer(training_preferences, validation_preferences)

    goal_examples_data = load_goal_frames("train", debug=False)

    global_step = 0
    epoch = 0
    complete = False
    iter_start_time = time.time()
    save_out = []
    losses = []
    plt.ion()

    print(f"Begin Training Loop with {len(training_preferences)} preferences!")

    # Main Training Loop
    eval_goal = trainer.calculate_goal_embedding(goal_examples_data)
    try:
        while not complete:
            for batch_i in range(0, len(training_preferences), XPREFS_CONFIG.irl.batch_size):

                # Recompute the Goal if needed
                if XPREFS_CONFIG.irl.recompute_goal_every is not None:
                    if global_step > 0 and global_step % XPREFS_CONFIG.irl.recompute_goal_every == 0:
                        eval_goal = trainer.calculate_goal_embedding(goal_examples_data)
                        print(f"Goal Recompute: {eval_goal}")

                # Run one batch through the network
                train_loss = trainer.train_xprefs_pair(batch_i, eval_goal)
                print(f"Training Loss for step {global_step}: {train_loss.item()}")

                if not global_step % XPREFS_CONFIG.irl.checkpointing_frequency:
                    trainer.save_checkpoint(global_step)

                if not global_step % XPREFS_CONFIG.irl.eval_every:
                    # eval_goal = eval_goal_embedding(model, goal_examples_data)
                    print("Running Validation Loop!")
                    test_loss, test_acc = trainer.validation_loop(eval_goal, train=False)
                    train_loss_whole_set, train_acc = trainer.validation_loop(eval_goal, train=True)
                    print(
                        "Iter[{}/{}] (Epoch {}), {:.6f}s/iter, Loss: {:.3f}, Test Loss: {:.3f}, Test Accuracy: {:3f}, Train Loss: {:.3f}, Train Accuracy: {:3f}".format(
                            global_step,
                            XPREFS_CONFIG.irl.train_max_iters,
                            epoch,
                            time.time() - iter_start_time,
                            train_loss.item(),
                            test_loss,
                            test_acc,
                            train_loss_whole_set,
                            train_acc
                        ))
                    save_out.append([global_step, epoch, train_loss.item(), test_loss, test_loss, train_loss_whole_set, train_acc])

                global_step += 1
                if global_step > XPREFS_CONFIG.irl.train_max_iters:
                    complete = True
                    break

                iter_start_time = time.time()

            epoch += 1

    except KeyboardInterrupt:
        print("Caught keyboard interrupt. Saving model before quitting.")

    finally:
        trainer.save_checkpoint(global_step)
        data_out = pd.DataFrame(save_out)
        data_out.columns = ["steps", "epochs", "train_loss", "test_loss", "test_acc", "training_loss", "training_acc"]
        data_out.to_csv(os.path.join(exp_dir, "embedding_train.csv"))
        np.savetxt(os.path.join(exp_dir, "goal_embedding.csv"), eval_goal.cpu().numpy(), delimiter=",")

    print("Training terminated.")


if __name__ == "__main__":
    train_xprefs()
