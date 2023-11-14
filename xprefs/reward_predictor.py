from torchkit import CheckpointManager

from xirl import factory
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import time
from xirl.models import Resnet18LinearEncoderNet
from xirl.models import ReinforcementLearningHumanFeedback

class XPrefsRewardTrainer:
    """
    Trains an embedding model based off preferences alone and returns predictions
    """
    def __init__(self, config, exp_dir, force_device=None):
        self.config = config
        self.exp_dir = exp_dir
        self.model_type = self.config.irl.learning_type
        self.device = self.load_device() if force_device is None else force_device
        self.model = self.initialize_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.irl.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.batch_size = config.irl.batch_size
        self.use_average_reward = config.irl.average_learned_reward
        self.load_checkpoint_manager()
        self.training_dataset = None
        self.validation_dataset = None
        self.training_preferences = None
        self.validation_preferences = None

    def train_xprefs_pair(self, i, goal_embedding):
        """
        Assumes o2 is preferred to o1
        """
        self.model.train()
        self.optimizer.zero_grad()

        reward_output_pairs = []

        # Batch the inputs to the network before calculating loss
        # TODO: Consider shuffling
        for j in range(i, i + self.batch_size):
            if j >= len(self.training_preferences):
                break

            o1, o2 = self.get_ith_from_preferences(self.training_preferences, self.training_dataset, j)
            assert not torch.equal(o1["frames"], o2["frames"])


            sum_reward_o1 = self.r_from_traj(o1, goal_embedding)
            sum_reward_o2 = self.r_from_traj(o2, goal_embedding)

            reward_output_pairs.append(torch.stack([sum_reward_o1, sum_reward_o2]))

        # Cross entropy loss over summed rewards
        loss = self.criterion(torch.stack(reward_output_pairs), torch.tensor([0 for _ in reward_output_pairs]).to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss

    def validate_r_from_traj(self, observation, goal_embedding):
        # self.model.eval()
        with torch.no_grad():
            r = self.r_from_traj(observation, goal_embedding)
        # self.model.train()
        return r

    def r_from_traj(self, observation, goal_embedding):
        o_frames = torch.stack([observation["frames"].to(self.device)])
        embed_o = self.model(o_frames).embs.squeeze()
        if self.model_type == "Xprefs":
            g_e = torch.squeeze(goal_embedding)
            g_e = g_e.repeat(len(embed_o), 1)
            goal_diff = g_e - embed_o
            dist_to_reward_o = torch.norm(goal_diff, dim=1)

            # Reward is the negative distance to goal
            sum_reward_o = -torch.sum(dist_to_reward_o)

            if self.use_average_reward:
                return sum_reward_o / len(embed_o)

            return sum_reward_o

        elif self.model_type == "RLHF":
            return embed_o

    def validation_loop(self, eval_goal, train=False):
        with torch.no_grad():
            cumulative_loss = 0.0
            total_correct, total_seen = 0, 0
            dataset = self.validation_dataset if not train else self.training_dataset
            prefs = self.validation_preferences if not train else self.training_preferences
            validation_loop_start = time.time()
            for j in range(len(prefs)):
                o1, o2 = self.get_ith_from_preferences(prefs, dataset, j)

                sum_reward_o1 = self.validate_r_from_traj(o1, eval_goal)
                sum_reward_o2 = self.validate_r_from_traj(o2, eval_goal)
                reward_out_pair = [sum_reward_o1, sum_reward_o2]

                loss = self.criterion(torch.stack(reward_out_pair).unsqueeze(0), torch.tensor([0]).to(self.device))
                cumulative_loss += loss.item()

                if sum_reward_o1.item() > sum_reward_o2.item():
                    total_correct += 1
                total_seen += 1

            return cumulative_loss / len(prefs), total_correct / total_seen, time.time() - validation_loop_start

    def calculate_goal_embedding(self, goal_dataloader):
        self.model.eval()
        with torch.no_grad():
            sum = None
            total_embeddings = 0
            for batch in goal_dataloader:
                frames = batch["frames"].to(self.device)
                out = self.model(frames).embs
                total_embeddings += len(out)
                if sum is None:
                    sum = torch.sum(out, dim=0)
                else:
                    sum += torch.sum(out, dim=0)
        self.model.train()
        # Average the sum of embeddings
        return sum / total_embeddings

    def load_checkpoint_manager(self):
        checkpoint_dir = os.path.join(self.exp_dir, "checkpoints")
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir,
            model=self.model,
        )
        global_step = self.checkpoint_manager.restore_or_initialize()
        return global_step

    def save_checkpoint(self, i):
        self.checkpoint_manager.save(i)

    def attach_data_to_trainer(self, train_data, valid_data):
        self.training_dataset = train_data
        self.validation_dataset = valid_data

    def attach_prefs_to_trainer(self, train_prefs, valid_prefs):
        self.training_preferences = train_prefs
        self.validation_preferences = valid_prefs

    def initialize_model(self):
        if self.model_type == "Xprefs":
            return Resnet18LinearEncoderNet(
                embedding_size=self.config.irl.embedding_size,
                num_ctx_frames=1,
                normalize_embeddings=False,
                learnable_temp=False,
            ).to(self.device)
        elif self.model_type == "RLHF":
            return Reinf(
                num_ctx_frames=1,
                normalize_embeddings=False,
                learnable_temp=False,
            ).to(self.device)
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