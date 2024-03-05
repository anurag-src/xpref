"""
TODO: Implement Bucket TCC

Goals:
1. Understand where TCC loss is being computed [DONE]
2. Understand how pairwise assignment in TCC dataloading works [DONE]
3. Parse Trajectories into buckets
4. Load Bucket-based data into TCC
"""

import os
import time
import torch
import yaml

from configs.xmagical.pretraining.tcc import get_config
from xirl import factory
from torchkit import CheckpointManager
from xirl.losses import compute_tcc_loss
import pandas as pd
from utils import setup_experiment, ConfigDict

# XIRL_CONFIG_FILE = "base_configs/pretrain.py"
CONFIG = get_config()
EXPERIMENT_DIRECTORY = os.path.expanduser("~/Documents/Xpref/experiments")

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

def load_dataset(split_type="train", debug=False):
    use_buckets = split_type == "train"
    dataset = factory.dataset_from_config(CONFIG, False, split_type, debug, with_reward=use_buckets)
    print("type: ", type(dataset))
    if use_buckets:
        print("Dataset built!")
        print(len(dataset.reward_set))

    if use_buckets:
        batch_sampler = factory.video_sampler_from_config(
            CONFIG, dataset.dir_tree, downstream=False, sequential=debug, rewards=dataset.reward_set)
    else:
        batch_sampler = factory.video_sampler_from_config(
            CONFIG, dataset.dir_tree, downstream=False, sequential=debug)

    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        batch_sampler=batch_sampler,
        num_workers=4 if torch.cuda.is_available() and not debug else 0,
        pin_memory=torch.cuda.is_available() and not debug,
    )

def load_model():
    return factory.model_from_config(CONFIG)

def load_tcc_optimizer(model):
    return factory.optim_from_config(config=CONFIG, model=model)

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
        name = "tcc_" + str(int(time.time()))
    exp_dir = os.path.join(EXPERIMENT_DIRECTORY, name)
    os.makedirs(exp_dir)
    with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
        yaml.dump(ConfigDict.to_dict(CONFIG), fp)
    return exp_dir

def train_one_iteration(model, optimizer, criterion, batch, device="cuda"):
    model.train()
    optimizer.zero_grad()

    frames = batch["frames"].to(device)
    out = model(frames)

    loss = criterion(out.embs, batch)
    loss.backward()
    optimizer.step()
    return loss

def train_bucket_tcc():
    device = load_device()
    print(f"Using {device} device")

    model = load_model().to(device)
    optimizer = load_tcc_optimizer(model)
    trainer = load_trainer(model, optimizer, device)
    exp_dir = create_experiment_dir()
    # setup_experiment(exp_dir, CONFIG, True)

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

    global_step = checkpoint_manager.restore_or_initialize()
    total_batches = max(1, len(batch_loaders["train"]))
    epoch = int(global_step / total_batches)
    complete = False

    criterion = tcc_loss
    iter_start_time = time.time()
    try:
        while not complete:
            for batch in batch_loaders["train"]:
                train_loss = train_one_iteration(model, optimizer, criterion, batch, device)

                if not global_step % CONFIG.checkpointing_frequency:
                    checkpoint_manager.save(global_step)

                global_step += 1
                if global_step > CONFIG.optim.train_max_iters:
                    complete = True
                    break

                print("Iter[{}/{}] (Epoch {}), {:.6f}s/iter, Loss: {:.3f}".format(
                global_step,
                    CONFIG.optim.train_max_iters,
                    epoch,
                    time.time() - iter_start_time,
                    train_loss.item(),
                ))
                iter_start_time = time.time()
            epoch += 1

    except KeyboardInterrupt:
        print("Caught keyboard interrupt. Saving model before quitting.")

    finally:
        checkpoint_manager.save(global_step)

    print("Training terminated.")

if __name__ == "__main__":
    train_bucket_tcc()