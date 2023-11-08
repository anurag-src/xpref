import torch
import numpy as np
from xirl.models import *
import pandas as pd
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt
import os
def preference_prob(r_hat, o1, o2,device):
    """
    Params:
    - Reward predictor 'r_hat' (Pytorch Network)
    - Obseration 1, 'o1', a finite sequence of frames (in this case, embedded by a netowrk phi)
    - Obseration 2, 'o2', a finite sequence of frames (in this case, embedded by a network phi)

    Return:
    The likelihood of observation 1 being preferred to observation 2
    """

    # TODO: Batch this input later, for now it's 1 frame at a time.
    sum_over_o1 = sum([r_hat.forward(torch.from_numpy(o1_t).float().to(device))for o1_t in o1])
    sum_over_o2 = sum([r_hat.forward(torch.from_numpy(o2_t).float().to(device)) for o2_t in o2])
    return torch.exp(sum_over_o1) / (torch.exp(sum_over_o1) + torch.exp(sum_over_o2))

def predictor_loss(r_hat, d_prefs):
    """
    Params:
    - Reward predictor 'r_hat' (Pytorch Network)
    - A batch of preferences, 'd_prefs', a python iterable of (o1, o2, pref) tuples, where,
        - o1 = Obseration 1, a finite sequence of frames
        - o2 = Obseration 2, a finite sequence of frames
        - pref = A binary preference in {0, 1}, where 0 indicated that o1 is preferred to o2 (o1 > o2) and 1 indicates the counter example.

    Return:
    A tensor loss object.

    NOTE: Backprop is not performed here. Instead the entire pytorch loss tensor is returned.
    """
    summed_loss = 0
    for o1, o2 in d_prefs:
        p = preference_prob(r_hat, o1, o2)
        summed_loss += torch.log(p)
    
    # Return the negative sum of log-probs for the set d_prefs
    return -summed_loss
def prepdata():
    df = pd.read_csv("embeddings.csv", header = None)
    embodiments = ["longstick", "mediumstick", "shortstick", "gripper"]
    #print(data.head())
    dataset = dict()
    for embodiment in embodiments:
        dataset[embodiment] = dict()
    df.columns = ["embodiment", "traj_id", "frame_num","embedding"]
    #print(df.columns)
    df1 = df.groupby(["embodiment", "traj_id"])
    #print(df1.head())
    for i,dfx in df1:
        #print(i)
        temp = []
        ##assumes frame numbers are conitnuous and in order
        for _,row in dfx.iterrows():
            temp.append(list(map(float, row[3][1:-1].split(', '))))
        
        dataset[row[0]][row[1]] = np.array(temp)
    return dataset
def preppref():
    pref_dataset_path = "/home/masters3/Documents/Research/Xpref/demos/random_demos_goal_compute/xmagical_0.2/train/"
    #filenames = ["gripper.csv","longstick.csv","mediumstick.csv","shortstick.csv"]
    filenames = ["longstick.csv", "mediumstick.csv", "shortstick.csv"]
    seed = 0
    np.random.seed(seed)  
    prefs = []
    for file in filenames:
        path = pref_dataset_path + file
        df = pd.read_csv(path,header = None)
        for _, row in df.iterrows():
            temp = []
            if row[2] != "mediumstick":
                for i in row:
                    temp.append(i)
                prefs.append(temp)      
    np.random.shuffle(prefs)
    return prefs[:2000]

def prepprefvalid():
    pref_dataset_path = "/home/masters3/Documents/Research/Xpref/demos/random_demos_goal_compute/xmagical_0.2/train/"
    #filenames = ["gripper.csv","longstick.csv","mediumstick.csv","shortstick.csv"]
    filenames = ["longstick.csv", "mediumstick.csv", "shortstick.csv"]
    seed = 0
    np.random.seed(seed)  
    prefs = []
    for file in filenames:
        path = pref_dataset_path + file
        df = pd.read_csv(path,header = None)
        for _, row in df.iterrows():
            temp = []
            if row[2] == "mediumstick":
                for i in row:
                    temp.append(i)
                prefs.append(temp)      
    np.random.shuffle(prefs)
    return prefs[:400]

def accuracy(r_hat,prefs,emb_dataset,device):
    count = 0
    for i in range(len(prefs)):
        traj1 = prefs[i][0]
        traj2 = prefs[i][1]
        emb = prefs[i][2]
        o1 = emb_dataset[emb][traj1]
        o2 = emb_dataset[emb][traj2]
        prob = preference_prob(r_hat,o1,o2,device)
        if prob > 0.5:
            count += 1
    return count/len(prefs)

def plot_lines(valid_loss_graph, train_loss_graph):
    valid_x = [point[0] for point in valid_loss_graph]
    valid_y = [point[1] for point in valid_loss_graph]
    
    train_x = [point[0] for point in train_loss_graph]
    train_y = [point[1] for point in train_loss_graph]
    
    plt.figure(figsize=(10, 6))
    plt.plot(valid_x, valid_y, label='Validation Loss', marker='o')
    plt.plot(train_x, train_y, label='Train Loss', marker='x')
    
    plt.title('Loss Graph')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--reward_model_path', default='', help="name and location for learned model params, e.g. ./learned_models/breakout.params")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.reward_model_path), exist_ok=True)
    PATH = args.reward_model_path + "model_"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    epochs = 100
    batch_size = 8
    train_test = 5
    emb_dataset = prepdata()
    lr = 0.0005
    weight_decay = 0.0
    r_hat = PreferenceRewardPredictor()
    r_hat.to(device)
    optimizer = optim.Adam(r_hat.parameters(),  lr=lr, weight_decay=weight_decay)


    #print(dataset["longstick"][826][41])
    #print(len(dataset["longstick"][826]))
    prefs = preppref()
    test = prepprefvalid()
    valid = test[:200]
    test = test[200:]
    pref_epoch = [i for i in range(len(prefs))]
    batch_loss = 0
    train_loss = 0
    train_loss_graph = []
    valid_loss_graph = []
    valid_accuracy_graph = []
    #print(len(prefs))
    #print(prefs)
    #print(preference_prob(r_hat,emb_dataset["longstick"][826],emb_dataset["longstick"][1685]))
    print("TRAINING")
    for epoch in range(epochs):
        optimizer.zero_grad()
        np.random.shuffle(pref_epoch)
        for i in range(len(prefs)):
            traj1 = prefs[pref_epoch[i]][0]
            traj2 = prefs[pref_epoch[i]][1]
            emb = prefs[pref_epoch[i]][2]
            o1 = emb_dataset[emb][traj1]
            o2 = emb_dataset[emb][traj2]
            l = torch.log(preference_prob(r_hat,o1,o2,device))
            batch_loss -= l
            train_loss -= l
            if (i + 1)%batch_size == 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                batch_loss = 0
        valid_loss = 0
        train_loss_graph.append([epoch,train_loss.item()/len(prefs)])
        for j in range(len(valid)):
            traj1 = valid[j][0]
            traj2 = valid[j][1]
            emb = valid[j][2]
            o1 = emb_dataset[emb][traj1]
            o2 = emb_dataset[emb][traj2]
            valid_loss -= torch.log(preference_prob(r_hat,o1,o2,device))
        valid_loss_graph.append([epoch,valid_loss.item()/len(prefs)])
        valid_accuracy = accuracy(r_hat,valid,emb_dataset,device)
        valid_accuracy_graph.append(valid_accuracy)
        print("epoch {}: training loss {}".format(epoch, train_loss.item()/len(prefs)))
        print("epoch {}: valid loss {}".format(epoch, valid_loss.item()/len(valid)))
        print("epoch {}: valid accuracy {}".format(epoch, valid_accuracy))
        train_loss = 0.0
        #print("check pointing")
        if (epoch+1)%25 == 0:
            torch.save({
        'epoch': epoch,
        'model_state_dict': r_hat.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': valid_loss,
        }, PATH + str(epoch) + ".pt")
    print("*********TRAINING DONE*********")
    print("Testing accuracy : ", accuracy(r_hat,test,emb_dataset,device))
    plot_lines(valid_loss_graph,train_loss_graph)
    #for _,row in df.iterrows():
        

        
