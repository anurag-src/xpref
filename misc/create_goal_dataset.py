"""
Given the XMagical set, create a dataset that represents the goal embeddings for each trajectory.
For this script, we extract ALL goal embeddings and save them to a goal dataset file.
The actual number of goals considered during experimentation can be varied within those experiments

USAGE: Fill in constants at the top of this file with your desired input/output directories, then run
    'python create_goal_dataset.py'
"""
import time
import os
import functools
import shutil

X_MAGICAL_DATA_PATH = "/home/connor/Documents/Xpref/xmagical"
OUTPUT_GOALSET_PATH = "/home/connor/Documents/Xpref/goal_examples"

def exists_dir_for_each(path_list):
    """
    Confirm that the directory exists for each path in path_list and is a directory
    """
    for p in path_list:
        if not os.path.exists(p):
            return False
        if not os.path.isdir(p):
            return False
    return True

def build_goalset_for_embedding(embodiment_name="gripper"):
    """
    Given an embedding name, e.g. "gripper", "longstick", "mediumstick", etc., extract all of the goal frame and place them in the path specified
    """
    f_in = [os.path.join(X_MAGICAL_DATA_PATH, mode, embodiment_name) for mode in ["train", "valid"]]
    f_out = [os.path.join(OUTPUT_GOALSET_PATH, mode, embodiment_name) for mode in ["train", "valid"]]

    # Check for valid input directories
    if not exists_dir_for_each(f_in):
        raise Exception(f"Provided path does not exist for one or more elements of f_in: {f_in}")

    # Create necessary output directories
    for o in f_out:
        if not os.path.exists(o):
            os.makedirs(o)

    for in_path, out_path in zip(f_in, f_out):
        folders = os.listdir(in_path)
        folders.sort()
        for d in folders:
            if not os.path.isdir(os.path.join(in_path, d)):
                continue
            dir_files = os.listdir(os.path.join(in_path, d))

            # Filter all non .png files, count number and subtract 1 for the name of the last file
            id_of_last_file = len(list(filter(lambda x: x.endswith(".png"), dir_files))) - 1
            file_name = f"{id_of_last_file}.png"
            dest = os.path.join(out_path, f"{d}.png")

            # Copy file to new directory
            shutil.copy(os.path.join(in_path, d, file_name), dest)


if __name__ == "__main__":
    """
    USAGE: Fill in constants at the top of this file with your desired input/output directories, then run
    'python create_goal_dataset.py'
    """
    start = time.time()
    print("Creating goal set...")
    for embodiment in ["gripper", "shortstick", "mediumstick", "longstick"]:
        print(f"Building goal dataset for {embodiment}...")
        build_goalset_for_embedding(embodiment_name=embodiment)
    print(f"Goal set created in {time.time() - start}s")
