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
from xirl import factory
from torchkit import CheckpointManager
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_collections import config_dict

ConfigDict = config_dict.ConfigDict

# XIRL_CONFIG_FILE = "base_configs/pretrain.py"
XPREFS_CONFIG = get_xprefs_config()


GOALSET_PATH = os.path.expanduser("~/Documents/Xpref/goal_examples")
LIM_GOALS_PER_EMBODIMENT = 200
EXPERIMENT_DIRECTORY = os.path.expanduser("~/Documents/Xpref/experiments")
# LOAD_CHECKPOINT = "/home/connor/Documents/Xpref/experiments/09-26-23-TCCMQME"
LOAD_CHECKPOINT = None

USE_AVERAGE_REWARD = True
PREFERENCES_FILE = os.path.expanduser("~/Documents/Xpref/trajectories2/train/cross_embedding_prefs.csv")
REMOVE_FROM_PREFERENCES = "mediumstick"

BATCH_SIZE = 20
EVAL_EVERY = 250



def get_ith_from_preferences(preferences, dataset, i):
    data_row = preferences.iloc[i]
    o1, o2, e1, e2 = int(data_row["o1_id"]), int(data_row["o2_id"]), data_row["o1_embod"], data_row["o2_embod"]
    return dataset.get_item(e1, o1), dataset.get_item(e2, o2)

def load_goal_frames(split_type="Train", debug=False):
    dataset = factory.goal_dataset_from_config(CONFIG, GOALSET_PATH, LIM_GOALS_PER_EMBODIMENT, False, split_type, debug)
    return torch.utils.data.DataLoader(
        dataset,
        # collate_fn=dataset.collate_fn,
        batch_size=50,
        num_workers=4 if torch.cuda.is_available() and not debug else 0,
        pin_memory=torch.cuda.is_available() and not debug,
    )


def create_experiment_dir(name=None):
    if not os.path.exists(EXPERIMENT_DIRECTORY):
        os.makedirs(EXPERIMENT_DIRECTORY)
    if name is None:
        name = str(int(time.time()))
    exp_dir = os.path.join(EXPERIMENT_DIRECTORY, name)
    os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
        yaml.dump(ConfigDict.to_dict(CONFIG), fp)
    return exp_dir

def eval_goal_embedding(model, goals, device="cuda"):
    with torch.no_grad():
        return calculate_goal_embedding(model, goals, eval=True, device=device)

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

def cumulative_r_from_traj(observation, model, goal_embedding, device, eval=False, average=False):
    if eval:
        model.eval()
        with torch.no_grad():
            o_frames = torch.stack([observation["frames"].to(device)])
            embed_o = model(o_frames).embs.squeeze()
            g_e = torch.squeeze(goal_embedding)
            g_e = g_e.repeat(len(embed_o), 1)
            goal_diff = g_e - embed_o
            dist_to_reward_o = torch.norm(goal_diff, dim=1)
            sum_reward_o = -torch.sum(dist_to_reward_o)
            if average:
                return sum_reward_o / len(embed_o)
            return sum_reward_o

    o_frames = torch.stack([observation["frames"].to(device)])
    embed_o = model(o_frames).embs.squeeze()
    g_e = torch.squeeze(goal_embedding)
    g_e = g_e.repeat(len(embed_o), 1)
    goal_diff = g_e - embed_o
    dist_to_reward_o = torch.norm(goal_diff, dim=1)

    # Reward is the negative distance to goal
    sum_reward_o = -torch.sum(dist_to_reward_o)
    if average:
        return sum_reward_o / len(embed_o)
    return sum_reward_o


def train_xprefs_pair(model, optimizer, i, preferences, dataset, goal_embedding, device="cuda", average=False):
    """
    Assumes o2 is preferred to o1
    """
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()

    reward_output_pairs = []

    # Batch the inputs to the network before calculating loss
    for j in range(i, i + BATCH_SIZE):
        if j >= len(preferences):
            break

        o1, o2 = get_ith_from_preferences(preferences, dataset, j)

        sum_reward_o1 = cumulative_r_from_traj(o1, model, goal_embedding, device, average=average)
        sum_reward_o2 = cumulative_r_from_traj(o2, model, goal_embedding, device, average=average)

        reward_output_pairs.append(torch.stack([sum_reward_o1, sum_reward_o2]))

    # Cross entropy loss over summed rewards
    loss = criterion(torch.stack(reward_output_pairs), torch.tensor([0 for _ in reward_output_pairs]).to(device))
    loss.backward()
    optimizer.step()
    return loss

def validation_xprefs(model, validation_prefs, dataset, eval_goal, device="cuda", average=False):
    print("Validating Test loss and accuracy...")
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    cumulative_loss = 0.0
    total_correct, total_seen = 0, 0
    for j in range(len(validation_prefs)):
        o1, o2 = get_ith_from_preferences(validation_prefs, dataset, j)

        sum_reward_o1 = cumulative_r_from_traj(o1, model, eval_goal, device, average=average)
        sum_reward_o2 = cumulative_r_from_traj(o2, model, eval_goal, device, average=average)
        reward_out_pair = [sum_reward_o1, sum_reward_o2]

        loss = criterion(torch.stack(reward_out_pair), torch.tensor(0).to(device))
        cumulative_loss += loss.item()

        if sum_reward_o1.item() > sum_reward_o2.item():
            total_correct += 1
        total_seen += 1

    return cumulative_loss / total_seen, total_correct / total_seen

def eval_one_iteration(model, criterion, validation_set, num_iters=None, device="cuda"):
    with torch.no_grad():
        model.eval()
        total_loss = 0
        it_ = 0
        for batch_idx, batch in enumerate(validation_set):
            if num_iters is not None and batch_idx >= num_iters:
                break
            frames = batch["frames"].to(device)
            out = model(frames)
            total_loss += criterion(out.embs, batch)
            it_ += 1
        return total_loss / it_

def train_xprefs():
    device = load_device()
    print(f"Using {device} device")

    model = load_model().to(device)
    optimizer = load_tcc_optimizer(model)
    if LOAD_CHECKPOINT is None:
        exp_dir = create_experiment_dir()
    else:
        exp_dir = LOAD_CHECKPOINT

    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir,
        model=model,
        optimizer=optimizer,
    )

    # Load the preference data from the config using a Dataloader, returns pandas Dataframe
    training_preferences = PreferenceLoader(XPREFS_CONFIG, train=True).preferences
    validation_preferences = PreferenceLoader(XPREFS_CONFIG, train=False).preferences
    training_trajectories = TrajectoryLoader.full_dataset_from_config(XPREFS_CONFIG, train=True)
    validation_trajectories = TrajectoryLoader.full_dataset_from_config(XPREFS_CONFIG, train=False)

    goal_examples_data = load_goal_frames("train", debug=False)

    global_step = checkpoint_manager.restore_or_initialize()
    # total_batches = max(1, len(batch_loaders["train"]))
    epoch = 0
    complete = False

    iter_start_time = time.time()
    save_out = []
    losses = []
    plt.ion()

    print(f"Begin Training Loop with {len(training_preferences)} preferences!")
    # Main Training Loop
    RECOMPUTE_GOAL_AT = []
    eval_goal = eval_goal_embedding(model, goal_examples_data, device=device)
    try:
        while not complete:
            for batch_i in range(0, len(training_preferences), BATCH_SIZE):
                train_loss = train_xprefs_pair(model, optimizer, batch_i, training_preferences, training_trajectories, eval_goal, device, average=USE_AVERAGE_REWARD)
                print(f"Training Loss for step {global_step}: {train_loss.item()}")

                if not global_step % CONFIG.checkpointing_frequency:
                    checkpoint_manager.save(global_step)

                if not global_step % EVAL_EVERY:
                    # eval_goal = eval_goal_embedding(model, goal_examples_data)
                    test_loss = validation_xprefs(model, validation_preferences, validation_trajectories, eval_goal, device=device, average=USE_AVERAGE_REWARD)
                    print("Iter[{}/{}] (Epoch {}), {:.6f}s/iter, Loss: {:.3f}, Test Loss: {:.3f}, Test Accuracy: {:3f}".format(
                        global_step,
                        CONFIG.optim.train_max_iters,
                        epoch,
                        time.time() - iter_start_time,
                        train_loss.item(),
                        test_loss[0],
                        test_loss[1],
                    ))
                    save_out.append([global_step, epoch, train_loss.item(), test_loss[0], test_loss[1]])

                if global_step > 0 and global_step in RECOMPUTE_GOAL_AT:
                    eval_goal = eval_goal_embedding(model, goal_examples_data)
                    print(f"Goal Recompute: {eval_goal}")

                global_step += 1
                if global_step > CONFIG.optim.train_max_iters:
                    complete = True
                    break

                iter_start_time = time.time()

            epoch += 1

    except KeyboardInterrupt:
        print("Caught keyboard interrupt. Saving model before quitting.")

    finally:
        checkpoint_manager.save(global_step)
        data_out = pd.DataFrame(save_out)
        data_out.columns = ["steps", "epochs", "train_loss", "test_loss", "test_acc"]
        data_out.to_csv(os.path.join(exp_dir, "embedding_train.csv"))
        np.savetxt(os.path.join(exp_dir, "goal_embedding.csv"), eval_goal.cpu().numpy(), delimiter=",")


    print("Training terminated.")

if __name__ == "__main__":
    train_xprefs()


















