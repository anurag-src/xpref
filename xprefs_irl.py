"""
In this file, we consider the XPrefs method where an average goal embedding is computed and used as the basis for a reward predictor
The model follows the same architecture as TCC.
"""
import os
import time
import torch
import yaml
from torchvision.datasets import ImageFolder
from configs.xmagical.pretraining.tcc import get_config
from xirl import factory
from torchkit import CheckpointManager
from xirl.losses import compute_tcc_loss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_collections import config_dict

ConfigDict = config_dict.ConfigDict

# XIRL_CONFIG_FILE = "base_configs/pretrain.py"
CONFIG = get_config()
X_MAGICAL_DATA_PATH = os.path.expanduser("~/Documents/Xpref/trajectories2")
# X_MAGICAL_DATA_PATH = os.path.expanduser("~/Documents/Xpref/xmagical")
CONFIG.data.root = X_MAGICAL_DATA_PATH

GOALSET_PATH = os.path.expanduser("~/Documents/Xpref/goal_examples")
LIM_GOALS_PER_EMBODIMENT = 10
EXPERIMENT_DIRECTORY = os.path.expanduser("~/Documents/Xpref/experiments")
# LOAD_CHECKPOINT = "/home/connor/Documents/Xpref/experiments/09-26-23-TCCMQME"
LOAD_CHECKPOINT = None

USE_AVERAGE_REWARD = True

if LOAD_CHECKPOINT:
    CONFIG.optim.train_max_iters = 5000
else:
    CONFIG.optim.train_max_iters = 10

TRAIN_EMBODIMENTS = tuple(["gripper", "shortstick", "longstick"])
CONFIG.data.pretrain_action_class = TRAIN_EMBODIMENTS
CONFIG.data.down_stream_action_class = TRAIN_EMBODIMENTS

PREFERENCES_FILE = os.path.expanduser("~/Documents/Xpref/trajectories2/train/cross_embedding_prefs.csv")
REMOVE_FROM_PREFERENCES = "mediumstick"

BATCH_SIZE = 15
# MAX_TRAINING_PREFS = 10000
MAX_TRAINING_PREFS = 100
MAX_TESTING_PREFS = 200
EVAL_EVERY = 50

def load_preferences(split_type="train"):
    df = pd.read_csv(PREFERENCES_FILE)
    # df.columns = ["o1_id", "o1_embod", "o1_reward", "o2_id", "o2_embod", "o2_reward"]
    # Remove the withheld embodiment from preference data
    df = df.loc[df["o1_embod"] != REMOVE_FROM_PREFERENCES]
    df = df.loc[df["o2_embod"] != REMOVE_FROM_PREFERENCES]
    if split_type == "train":
        df = df.head(MAX_TRAINING_PREFS)
    elif split_type == "valid":
        df = df.tail(MAX_TESTING_PREFS)
    return df

def load_device():
    # device = (
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else "cpu"
    # )
    device = "cpu"
    return device

def load_dataset(split_type="train", debug=False):
    dataset = factory.dataset_from_config(CONFIG, False, split_type, debug)
    batch_sampler = factory.video_sampler_from_config(
        CONFIG, dataset.dir_tree, downstream=False, sequential=debug)
    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        batch_sampler=batch_sampler,
        num_workers=4 if torch.cuda.is_available() and not debug else 0,
        pin_memory=torch.cuda.is_available() and not debug,
    )

def load_preference_dataset(split_type="train", debug=False):
    dataset = factory.full_dataset_from_config(CONFIG, False, split_type, debug)
    return dataset

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

def load_model():
    return factory.model_from_config(CONFIG)

def load_tcc_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=1e-3)
    # return factory.optim_from_config(config=CONFIG, model=model)

def load_trainer(model, optimizer, device):
    return factory.trainer_from_config(CONFIG, model, optimizer, device)

def tcc_loss(embs, batch):
    steps = batch["frame_idxs"].to(load_device())
    seq_lens = batch["video_len"].to(load_device())

    # Dynamically determine the number of cycles if using stochastic
    # matching.
    batch_size, num_cc_frames = embs.shape[:2]
    num_cycles = int(batch_size * num_cc_frames)

    config = CONFIG
    return compute_tcc_loss(
        embs=embs,
        idxs=steps,
        seq_lens=seq_lens,
        stochastic_matching=config.loss.tcc.stochastic_matching,
        normalize_embeddings=config.model.normalize_embeddings,
        loss_type=config.loss.tcc.loss_type,
        similarity_type=config.loss.tcc.similarity_type,
        num_cycles=num_cycles,
        cycle_length=config.loss.tcc.cycle_length,
        temperature=config.loss.tcc.softmax_temperature,
        label_smoothing=config.loss.tcc.label_smoothing,
        variance_lambda=config.loss.tcc.variance_lambda,
        huber_delta=config.loss.tcc.huber_delta,
        normalize_indices=config.loss.tcc.normalize_indices,
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

def train_one_iteration(model, optimizer, criterion, batch, device="cuda"):
    model.train()
    optimizer.zero_grad()

    frames = batch["frames"].to(device)
    out = model(frames)

    loss = criterion(out.embs, batch)
    loss.backward()
    optimizer.step()
    return loss

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
            sum_reward_o = torch.sum(dist_to_reward_o)
            if average:
                return sum_reward_o / len(embed_o)
            return sum_reward_o

    o_frames = torch.stack([observation["frames"].to(device)])
    embed_o = model(o_frames).embs.squeeze()
    g_e = torch.squeeze(goal_embedding)
    g_e = g_e.repeat(len(embed_o), 1)
    goal_diff = g_e - embed_o
    dist_to_reward_o = torch.norm(goal_diff, dim=1)
    sum_reward_o = torch.sum(dist_to_reward_o)
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

        if sum_reward_o1 > sum_reward_o2:
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

def train_tcc():
    device = load_device()
    print(f"Using {device} device")

    model = load_model().to(device)
    optimizer = load_tcc_optimizer(model)
    trainer = load_trainer(model, optimizer, device)
    exp_dir = create_experiment_dir()

    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir,
        model=model,
        optimizer=optimizer,
    )

    batch_loaders = {
        "train": load_dataset("train", debug=False),
        "valid": load_dataset("valid", debug=False)
    }
    goal_examples_data = load_goal_frames("train", debug=False)

    global_step = checkpoint_manager.restore_or_initialize()
    total_batches = max(1, len(batch_loaders["train"]))
    epoch = int(global_step / total_batches)
    complete = False

    criterion = tcc_loss
    iter_start_time = time.time()

    # Main Training Loop
    try:
        while not complete:
            for batch in batch_loaders["train"]:
                train_loss = train_one_iteration(model, optimizer, criterion, batch, device)
                eval_goal = eval_goal_embedding(model, goal_examples_data, device=device)

                if not global_step % CONFIG.checkpointing_frequency:
                    checkpoint_manager.save(global_step)

                if not global_step % CONFIG.eval.eval_frequency:
                    test_loss = eval_one_iteration(model, criterion=criterion, validation_set=batch_loaders["valid"], num_iters=10, device=device)
                    print("Iter[{}/{}] (Epoch {}), {:.6f}s/iter, Loss: {:.3f}, Test {:.3f}".format(
                        global_step,
                        CONFIG.optim.train_max_iters,
                        epoch,
                        time.time() - iter_start_time,
                        train_loss.item(),
                        test_loss.item()
                    ))

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

    print("Training terminated.")

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

    preferences = load_preferences()
    full_traj_dataset = load_preference_dataset("train", debug=False)
    valid_preferences_dataset = load_preferences("valid")

    goal_examples_data = load_goal_frames("train", debug=False)

    global_step = checkpoint_manager.restore_or_initialize()
    # total_batches = max(1, len(batch_loaders["train"]))
    epoch = 0
    complete = False

    criterion = tcc_loss
    iter_start_time = time.time()

    save_out = []

    losses = []
    plt.ion()

    print(f"Begin Training Loop with {len(preferences)} preferences!")
    # Main Training Loop
    RECOMPUTE_GOAL_AT = []
    eval_goal = eval_goal_embedding(model, goal_examples_data, device=device)
    try:
        while not complete:
            for batch_i in range(0, len(preferences), BATCH_SIZE):
                train_loss = train_xprefs_pair(model, optimizer, batch_i, preferences, full_traj_dataset, eval_goal, device, average=USE_AVERAGE_REWARD)
                print(f"Training Loss for step {global_step}: {train_loss.item()}")
                # losses.append(train_loss.item())

                # plt.plot(losses)
                # plt.pause(0.01)
                # plt.title("XPrefs Training Loss")
                # plt.ylabel("Loss")
                # plt.xlabel("Batch Updates")
                # plt.show()

                if not global_step % CONFIG.checkpointing_frequency:
                    checkpoint_manager.save(global_step)

                if not global_step % EVAL_EVERY:
                    # eval_goal = eval_goal_embedding(model, goal_examples_data)
                    test_loss = validation_xprefs(model, valid_preferences_dataset, full_traj_dataset, eval_goal, device=device, average=USE_AVERAGE_REWARD)
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
        np.savetxt(os.path.join(exp_dir, "goal_embedding.csv"), eval_goal.numpy(), delimiter=",")


    print("Training terminated.")

if __name__ == "__main__":
    train_xprefs()


















