import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xprefs.pref_loader import PreferenceLoader
from base_configs.xprefs import get_config as get_xprefs_config

TRAINING_RESULTS = "/home/connor/Documents/Xpref/experiments/traj1_11-13-23/embedding_train.csv"

def plot_accuracy():
    df = pd.read_csv(TRAINING_RESULTS)

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')

    plt.plot(df["steps"], df["test_acc"], label="Testing Data")
    plt.plot(df["steps"], df["training_acc"], label="Training Data")

    plt.title("IRL Accuracy")
    plt.legend()
    plt.show()

def plot_loss():
    df = pd.read_csv(TRAINING_RESULTS)

    plt.xlabel('Time (Epochs)')
    plt.ylabel('Dataset Loss')

    plt.plot(df["steps"], df["test_loss"], label="Testing Data")
    plt.plot(df["steps"], df["training_loss"], label="Training Data")

    plt.title("IRL Validation Loss")
    plt.legend()
    plt.show()

def plot_instant_loss():
    df = pd.read_csv(TRAINING_RESULTS)

    plt.xlabel('Time (Epochs)')
    plt.ylabel('Loss')

    plt.plot(df["steps"], df["train_loss"])

    plt.title("IRL Training Loss")
    plt.legend()
    plt.show()

def plot_training_distribution_difference():
    config = get_xprefs_config()
    training_preferences = PreferenceLoader(config, train=True).preferences
    diff = training_preferences["o1_reward"] - training_preferences["o2_reward"]
    count = len(diff[diff == 0])
    print("Zero difference!", count)
    plt.hist(diff, bins=30)
    plt.ylabel("Count")
    plt.xlabel("Difference in GT Reward Between Demo1 and Demo2")
    plt.show()

def plot_training_distribution():
    config = get_xprefs_config()
    training_preferences = PreferenceLoader(config, train=True).preferences
    diff = training_preferences["o1_reward"]
    plt.hist(diff, bins=20)
    plt.show()

if __name__ == "__main__":
    plot_accuracy()
    plot_loss()
    plot_instant_loss()
    # plot_training_distribution_difference()