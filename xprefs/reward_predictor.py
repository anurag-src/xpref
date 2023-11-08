from xirl import factory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from xirl.models import Resnet18LinearEncoderNet

class XPrefsRewardTrainer:
    """
    Trains an embedding model based off preferences alone and returns predictions
    """
    def __init__(self, config, force_device=None):
        self.config = config
        self.model_type = self.config.irl.learning_type
        self.model = self.initialize_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.irl.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = self.load_device() if force_device is None else force_device
        self.batch_size = config.irl.batch_size
        self.use_average_reward = config.irl.average_learned_reward
        self.training_dataset = None
        self.validation_dataset = None
        self.training_preferences = None
        self.validation_preferences = None

    def train_xprefs_pair(self, optimizer, i, preferences, dataset, goal_embedding,):
        """
        Assumes o2 is preferred to o1
        """
        self.model.train()
        optimizer.zero_grad()

        reward_output_pairs = []

        # Batch the inputs to the network before calculating loss
        for j in range(i, i + self.batch_size):
            if j >= len(preferences):
                break

            o1, o2 = self.get_ith_from_preferences(preferences, dataset, j)

            sum_reward_o1 = self.r_from_traj(o1, goal_embedding)
            sum_reward_o2 = self.r_from_traj(o2, goal_embedding)

            reward_output_pairs.append(torch.stack([sum_reward_o1, sum_reward_o2]))

        # Cross entropy loss over summed rewards
        loss = criterion(torch.stack(reward_output_pairs), torch.tensor([0 for _ in reward_output_pairs]).to(device))
        loss.backward()
        optimizer.step()
        return loss

    def r_from_traj(self, observation, goal_embedding, train=True):
        if not train:
            self.model.eval()
            with torch.no_grad():
                o_frames = torch.stack([observation["frames"].to(self.device)])
                embed_o = self.model(o_frames).embs.squeeze()
                g_e = torch.squeeze(goal_embedding)
                g_e = g_e.repeat(len(embed_o), 1)
                goal_diff = g_e - embed_o
                dist_to_reward_o = torch.norm(goal_diff, dim=1)
                sum_reward_o = -torch.sum(dist_to_reward_o)
                if average:
                    return sum_reward_o / len(embed_o)
                return sum_reward_o

        o_frames = torch.stack([observation["frames"].to(self.device)])
        embed_o = self.model(o_frames).embs.squeeze()
        g_e = torch.squeeze(goal_embedding)
        g_e = g_e.repeat(len(embed_o), 1)
        goal_diff = g_e - embed_o
        dist_to_reward_o = torch.norm(goal_diff, dim=1)

        # Reward is the negative distance to goal
        sum_reward_o = -torch.sum(dist_to_reward_o)
        if average:
            return sum_reward_o / len(embed_o)
        return sum_reward_o

    def attach_data_to_trainer(self, train_data, valid_data):
        self.training_dataset = train_data
        self.validation_dataset = valid_data

    def attach_prefs_to_trainer(self, train_prefs, valid_prefs):
        self.training_preferences = train_prefs
        self.validation_preferences = valid_prefs

    def initialize_model(self):
        if self.model_type == "Xprefs":
            return Resnet18LinearEncoderNet(embedding_size=self.config.irl.embedding_size)
        elif self.model_type == "RLHF":
            raise NotImplementedError("Need to implement end-to-end network for RLHF")
        else:
            raise Exception(f"Unknown Model type: {self.model_type}")

    def load_device(self):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        # device = "cpu"
        return device

    def get_ith_from_preferences(self, preferences, dataset, i):
        data_row = preferences.iloc[i]
        o1, o2, e1, e2 = int(data_row["o1_id"]), int(data_row["o2_id"]), data_row["o1_embod"], data_row["o2_embod"]
        return dataset.get_item(e1, o1), dataset.get_item(e2, o2)

    def _check_for_prefs_and_data(self):
        assert self.training_dataset is not None
        assert self.validation_dataset is not None
        assert self.training_preferences is not None
        assert self.validation_preferences is not None