import csv
import glob
import os
import argparse
import numpy as np
import random
from torchkit import CheckpointManager
from xirl import common
import utils
import glob
from tqdm.auto import tqdm
from xirl import factory
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import tensorflow as tf

def setup(exp_path):
    """
    Load the latest embedder checkpoint and dataloaders.
    (Connor) -- Modified Directly from XIRL Code
    """
    config = utils.load_config_from_dir(exp_path)
    model = common.get_model(config)
    downstream_loaders = common.get_downstream_dataloaders(config, False)["train"]
    checkpoint_dir = os.path.join(exp_path, "checkpoints")

    checkpoint_manager = CheckpointManager(checkpoint_dir, model=model)
    global_step = checkpoint_manager.restore_or_initialize()
    print("Restored model from checkpoint %d.", global_step)
    return model, downstream_loaders

def embed(model,downstream_loader,device,):
    """Embed the stored trajectories and compute mean goal embedding."""
    outs = []
    for class_name, class_loader in downstream_loader.items():
        print("Embedding %s.", class_name)
        for batch in tqdm(iter(class_loader), leave=False):
            out = model.infer(batch["frames"].to(device)) 
            #print(out)
            outs.append(out.embs)
    return outs


if __name__ == "__main__":
    experiment_name = "random_demos_goal_compute_0.5"
    model, loaders = setup(os.path.expanduser(f"~/Documents/Research/Xpref/pretrain_runs/random_goal_compute/{experiment_name}"))
    PARENT_DIR = "/home/masters3/Documents/Research/Xpref/demos/random_demos_goal_compute/xmagical_0.5/train/gripper"
    #path = PARENT_DIR + "/*.png"
    _img_to_tensor = transforms.ToTensor()
    writer = csv.writer(open('gripper_embeddings.csv','w'))
    #for embodiment in os.listdir(PARENT_DIR):
    #    if embodiment != ".DS_Store":
    #        print(embodiment)
    #        traj_folder_path = PARENT_DIR + str(embodiment)
    #        for traj in os.listdir(traj_folder_path):
    #            if traj != ".DS_Store":
    #                frames = []
    #                print(traj)
    #                path = traj_folder_path + "/" + str(traj) + '/*.png'
                    #print(path)
    #                for frame in glob.glob(path):
    #                    image = Image.open(os.path.join(PARENT_DIR, frame))
    #                    frames.append(torch.unsqueeze(_img_to_tensor(image), dim=0))
    #                t = torch.cat(frames, dim=0)
    #                out = model.infer(torch.unsqueeze(t, dim=0))
    #                tensor_list = tf.unstack(out.embs)
    #                for i in range(len(tensor_list)):
    #                    data = []
    #                    data.append(embodiment)
    #                    data.append(traj)
    #                    data.append(i)
    #                    data.append(str(tensor_list[i].numpy().tolist()))
    #                    writer.writerow(data)
    traj_folder_path = PARENT_DIR
    for traj in os.listdir(traj_folder_path):
                if traj != ".DS_Store":
                    frames = []
                    print(traj)
                    path = traj_folder_path + "/" + str(traj) + '/*.png'
                    #print(path)
                    for frame in glob.glob(path):
                        image = Image.open(os.path.join(PARENT_DIR, frame))
                        frames.append(torch.unsqueeze(_img_to_tensor(image), dim=0))
                    t = torch.cat(frames, dim=0)
                    out = model.infer(torch.unsqueeze(t, dim=0))
                    tensor_list = tf.unstack(out.embs)
                    for i in range(len(tensor_list)):
                        data = []
                        data.append("gripper")
                        data.append(traj)
                        data.append(i)
                        data.append(str(tensor_list[i].numpy().tolist()))
                        writer.writerow(data)

    #print(out.embs.shape)
    
