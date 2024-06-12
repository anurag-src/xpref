# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Video samplers for mini-batch creation."""

import abc
from typing import Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler

ClassIdxVideoIdx = Tuple[int, int]
DirTreeIndices = List[List[ClassIdxVideoIdx]]
VideoBatchIter = Iterator[List[ClassIdxVideoIdx]]


class VideoBatchSampler(abc.ABC, Sampler):
    """Base class for all video samplers."""

    def __init__(
            self,
            dir_tree,
            batch_size,
            sequential=False,
    ):
        """Constructor.

    Args:
      dir_tree: The directory tree of a `datasets.VideoDataset`.
      batch_size: The number of videos in a batch.
      sequential: Set to `True` to disable any shuffling or randomness.
    """
        assert isinstance(batch_size, int)

        self._batch_size = batch_size
        self._dir_tree = dir_tree
        self._sequential = sequential

    @abc.abstractmethod
    def _generate_indices(self):
        """Generate batch chunks containing (class idx, video_idx) tuples."""
        pass

    def __iter__(self):
        idxs = self._generate_indices()
        if self._sequential:
            return iter(idxs)
        return iter(idxs[i] for i in torch.randperm(len(idxs)))

    def __len__(self):
        num_vids = 0
        for vids in self._dir_tree.values():
            num_vids += len(vids)
        return num_vids // self.batch_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def dir_tree(self):
        return self._dir_tree


class RandomBatchSampler(VideoBatchSampler):
    """Randomly samples videos from different classes into the same batch.

  Note the `sequential` arg is disabled here.
  """

    def _generate_indices(self):
        # Generate a list of video indices for every class.
        all_idxs = []
        for k, v in enumerate(self._dir_tree.values()):
            seq = list(range(len(v)))
            all_idxs.extend([(k, s) for s in seq])
        # Shuffle the indices.
        all_idxs = [all_idxs[i] for i in torch.randperm(len(all_idxs))]
        # If we have less total videos than the batch size, we pad with clones
        # until we reach a length of batch_size.
        if len(all_idxs) < self._batch_size:
            while len(all_idxs) < self._batch_size:
                all_idxs.append(all_idxs[np.random.randint(0, len(all_idxs))])
        # Split the list of indices into chunks of len `batch_size`.
        idxs = []
        end = self._batch_size * (len(all_idxs) // self._batch_size)
        for i in range(0, end, self._batch_size):
            batch_idxs = all_idxs[i:i + self._batch_size]
            idxs.append(batch_idxs)

        # print("Video Sampler", idxs[:10])
        return idxs


class SameClassBatchSampler(VideoBatchSampler):
    """Ensures all videos in a batch belong to the same class."""

    def _generate_indices(self):
        idxs = []
        for k, v in enumerate(self._dir_tree.values()):
            # Generate a list of indices for every video in the class.
            len_v = len(v)
            seq = list(range(len_v))
            if not self._sequential:
                seq = [seq[i] for i in torch.randperm(len(seq))]
            # Split the list of indices into chunks of len `batch_size`,
            # ensuring we drop the last chunk if it is not of adequate length.
            batch_idxs = []
            end = self._batch_size * (len_v // self._batch_size)
            for i in range(0, end, self._batch_size):
                xs = seq[i:i + self._batch_size]
                # Add the class index to the video index.
                xs = [(k, x) for x in xs]
                batch_idxs.append(xs)
            idxs.extend(batch_idxs)
        return idxs


class SameClassBatchSamplerDownstream(SameClassBatchSampler):
    """A same class batch sampler with a batch size of 1.

  This batch sampler is used for downstream datasets. Since such datasets
  typically load a variable number of frames per video, we are forced to use
  a batch size of 1.
  """

    def __init__(
        self,
        dir_tree,
        sequential=False,
    ):
        super().__init__(dir_tree, batch_size=1, sequential=sequential)



class SameQualityBatchSampler(VideoBatchSampler):
    def __init__(self, dir_tree, batch_size, rewards, num_buckets=4, sequential=False):
        super().__init__(dir_tree, batch_size, sequential)

        """
        Assumes an iterable "rewards" of (reward, embodiment, video_index) triples, sorted by increasing reward
        """

        self.rewards = rewards
        self.n_buckets = num_buckets
        print("Length of Reward Sequence: ", len(self.rewards))

    def _generate_indices(self):
        idxs = []
        r_batches = []

        end = self._batch_size * (len(self.rewards) // self._batch_size)
        elements_per_bucket = len(self.rewards) // self.n_buckets
        batches_per_bucket = elements_per_bucket // self._batch_size
        for i in range(0, end, elements_per_bucket):
            max_batch = []
            # Create the largest possible batch size
            for j in range(elements_per_bucket):
                max_batch.append(self.rewards[i + j])

            # Sample from the bucket to create a batch of size _batch_size
            for b in range(batches_per_bucket):
                rew_batch = []
                batch = []

                idx = np.random.randint(0, len(max_batch), self._batch_size)
                for i in idx:
                    r, e, ind = max_batch[i]
                    batch.append((e, ind))
                    rew_batch.append(r)

                idxs.append(batch)
                r_batches.append(rew_batch)
        print("NEW VID SAMPLER INDICES: ", r_batches[-2:])
        return idxs


class TripletBatchSampler(VideoBatchSampler):
    def __init__(self, dir_tree, batch_size, rewards, sequential=False):
        super().__init__(dir_tree, batch_size, sequential)

        """
        Assumes an iterable "rewards" of (reward, embodiment, video_index) triples, sorted by increasing reward
        """

        self.rewards = rewards
        print("Length of Reward Sequence: ", len(self.rewards))

    def _generate_indices(self):
        """
        Outputs Batch of size (3B) x 3, where every row is the triplet (reward, embodiment, video_index)
        Every 3 rows are ordered in terms of decreasing reward. A > B > C
        """
        batches = []
        end = self._batch_size * (len(self.rewards) // self._batch_size)

        for i in range(0, end, self._batch_size):
            idxs = []
            rand_selection = np.random.randint(0, len(self.rewards), (self._batch_size, 3))
            for i, j, k in rand_selection:
                # Prevent duplicate trajectories in triplet
                if i == j: j -= 1
                if i == k: k -= 1
                if j == k: k -= 1
                assert i != j and i != k and j != k

                # Append the ith, jth, and kth rewards to the row starting with the highest reward and decreasing to the lowest reward.
                # This is some novice looking code right here but it works :)
                l = [i, j, k]
                l.sort()
                A, B, C = self.rewards[l[0]],  self.rewards[l[1]], self.rewards[l[2]]
                row = [
                    (A[1], A[2]),
                    (B[1], B[2]),
                    (C[1], C[2])
                ]
                idxs += row
            batches.append(idxs)
        return batches