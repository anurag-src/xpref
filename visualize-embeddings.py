from sklearn.manifold import TSNE
from torchkit import CheckpointManager
from xirl import common
import utils
import os
from tqdm.auto import tqdm
from xirl import factory
import torch
import numpy as np
import matplotlib.pyplot as plt

def setup(exp_path):
    """
    Load the latest embedder checkpoint and dataloaders.
    (Connor) -- Modified Directly from XIRL Code
    """
    config = utils.load_config_from_dir(exp_path)
    model = common.get_model(config)
    downstream_loaders = common.get_downstream_dataloaders(config, False)["valid"]
    checkpoint_dir = os.path.join(exp_path, "checkpoints")

    checkpoint_manager = CheckpointManager(checkpoint_dir, model=model)
    global_step = checkpoint_manager.restore_or_initialize()
    print("Restored model from checkpoint %d.", global_step)
    return model, downstream_loaders

def embed(
    model,
    downstream_loader,
    device,
):
    """Embed the stored trajectories and compute mean goal embedding."""
    outs = []
    for class_name, class_loader in downstream_loader.items():
        print("Embedding %s.", class_name)
        for batch in tqdm(iter(class_loader), leave=False):
            out = model.infer(batch["frames"].to(device))
            outs.append(out.embs)
    return outs

def visualize():
    pass

if __name__ == "__main__":
    device = "cpu"
    experiment_name = "good_demos_set"
    model, loaders = setup(os.path.expanduser(f"~/Documents/xprefs/pretrain_runs/{experiment_name}"))
    model.to(device).eval()
    outs_A = embed(model, loaders, device)
    print(outs_A)
    print("NORMAL OUTS DONE!")

    device = "cpu"
    experiment_name = "bad_demos_set"
    model, loaders = setup(os.path.expanduser(f"~/Documents/xprefs/pretrain_runs/{experiment_name}"))
    model.to(device).eval()
    outs_B = embed(model, loaders, device)
    print(outs_B)
    print("BAD DEMOS DONE!")

    EMBODIMENTS = ["gripper", "longstick", "mediumstick", "shortstick"]

    A_embed_frames = []
    A_class = []
    A_traj = []
    A_embodiments = []
    A_time = []

    print("Start A Embeddings")
    for i, embedding in enumerate(outs_A):
        _class = i % 4
        for j, frame in enumerate(embedding):
            A_embed_frames.append(frame.tolist())
            A_traj.append(i)
            A_class.append(1)
            A_embodiments.append(_class)
            A_time.append(j)

    print("A Embeddings Done!")
    print(len(A_embed_frames))

    B_embed_frames = []
    B_class = []
    B_traj = []
    B_embodiments = []
    B_time = []
    print("Start B Embeddings")
    for i, embedding in enumerate(outs_B):
        _class = i % 4
        for j, frame in enumerate(embedding):
            B_embed_frames.append(frame.tolist())
            B_traj.append(i)
            B_class.append(0)
            B_embodiments.append(_class)
            B_time.append(j)
    print("B Embeddings Done!")

    all_embeddings = A_embed_frames + B_embed_frames
    all_classes = A_class + B_class
    all_traj = A_traj + B_traj
    all_embodiments = A_embodiments + B_embodiments
    all_time = A_time + B_time

    print(all_embeddings)

    all_embeddings_np = np.array(all_embeddings)
    embed_to_2d = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30, method="exact").fit_transform(all_embeddings_np)
    print(embed_to_2d)

    # colors = [(0, 1 - (0.1 * all_embodiments[i]), 0) if all_classes[i] == 1 else (1 - (0.1 * all_embodiments[i]), 0, 0) for i in range(len(all_classes))]
    g_colors = [(0.0, min(0.2 + (0.01 * all_time[i]), 1.0), 0.0) for i in range(len(A_time))]
    r_colors = [(min(0.2 + (0.01 * all_time[i]), 1.0), 0.0, 0.0) for i in range(len(B_time))]
    colors = g_colors + r_colors

    plt.scatter(embed_to_2d[:,0], embed_to_2d[:,1], c=colors)
    plt.title("Embedded Trajectories in t-SNE 2D")
    plt.xlabel("t-SNE X")
    plt.ylabel("t-SNE Y")
    plt.show()

    print(":)")

