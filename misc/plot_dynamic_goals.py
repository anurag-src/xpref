import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

methods = [
    ("Goal_Recompute-4", "Dynamic (4 steps)"),
    ("Goal_Recompute-8", "Dynamic (8 steps)"),
    ("Goal_Recompute-400", "Dynamic (400 steps)"),
    ("Goal_Recompute-1000", "Dynamic (1000 steps)"),
    ("Xprefs", "Static")
]

def load_data(method):
    df = pd.read_csv(os.path.join('../data/Static_Dynamic_Graph_Data', method, "embedding_train.csv"))
    return df

if __name__ == "__main__":
    fig, ax = plt.subplots(1)
    for m, n in methods:
        df = load_data(m)
        # print(df)
        x = df["steps"]
        y = df["training_loss"]
        ax.plot(x, y, label=n)
        # ax.fill_between(x, y + err, y - err, alpha=0.15)

    plt.ylabel("Training Loss")
    plt.xlabel("Reward Learning Training Steps")
    # plt.title("")
    fig.tight_layout()
    plt.legend()
    plt.savefig("dynamic-comparison.pdf", format="pdf", bbox_inches="tight")
    plt.show()