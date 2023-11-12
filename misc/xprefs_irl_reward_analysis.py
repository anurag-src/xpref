import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAINING_RESULTS = "/home/connor/Documents/Xpref/experiments/toy_e1_trial1_mediumstick_only_static_goal_cumulative_reward/embedding_train.csv"

def plot_accuracy():
    df = pd.read_csv(TRAINING_RESULTS)

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')

    plt.plot(df["epochs"], df["test_acc"], label="Testing Data")
    plt.plot(df["epochs"], df["training_acc"], label="Training Data")

    plt.title("IRL Accuracy")
    plt.legend()
    plt.show()

def plot_loss():
    df = pd.read_csv(TRAINING_RESULTS)

    plt.xlabel('Time (Epochs)')
    plt.ylabel('Dataset Loss')

    plt.plot(df["epochs"], df["test_loss"], label="Testing Data")
    plt.plot(df["epochs"], df["training_loss"], label="Training Data")

    plt.title("IRL Validation Loss")
    plt.legend()
    plt.show()

def plot_instant_loss():
    df = pd.read_csv(TRAINING_RESULTS)

    plt.xlabel('Time (Epochs)')
    plt.ylabel('Loss')

    plt.plot(df["epochs"], df["train_loss"])

    plt.title("IRL Training Loss")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_accuracy()
    plot_loss()
    plot_instant_loss()