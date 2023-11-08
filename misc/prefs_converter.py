import numpy as np
import csv
import pandas as pd

FILE = "/home/connor/Documents/Xpref/trajectories/valid/same_embedding_prefs.csv"
MAX_PREFERENCES = 100_000

if __name__ == "__main__":

    print(f"Preparing to Read Data from file {FILE}")

    out_rows = []
    with open(FILE, 'r') as f:
        reader = csv.reader(f)
        row_i = 0
        for row in reader:
            o1 = list(eval(row[0]))
            id_1 = int(o1[0])
            embodiment_1 = o1[1]
            reward_1 = o1[2]

            o2 = list(eval(row[1]))
            id_2 = int(o2[0])
            embodiment_2 = o2[1]
            reward_2 = o2[2]

            out_rows.append([id_1, embodiment_1, reward_1, id_2, embodiment_2, reward_2])
            row_i += 1

    data = np.array(out_rows)
    np.random.shuffle(data)

    df = pd.DataFrame(data[:MAX_PREFERENCES])
    df.columns = ["o1_id", "o1_embod", "o1_reward", "o2_id", "o2_embod", "o2_reward"]
    df.to_csv(f"{FILE}_new")
