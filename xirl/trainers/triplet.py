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

"""Goal classifier trainer."""

from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from xirl.trainers.base import Trainer

BatchType = Dict[str, Union[torch.Tensor, List[str]]]


class TripletTrainer(Trainer):
  """
  A trainer that learns to represent rewards with the bradley-terry model via triplet embeddings
  """

  def tensor_contains_nan(self, tensor):
    return torch.isnan(tensor).any()

  def compute_loss(
      self,
      embs,
      batch,
  ):
    del batch

    batch_size = embs.shape[0] // 3
    embs = torch.reshape(embs, (batch_size, 3, self._config.frame_sampler.num_frames_per_sequence, self._config.model.embedding_size))

    # At this point, the tensor shape should be (N x 3 x num_cc_frames x emb_size)
    # Compute the distance function d for d(embs[A], embs[B]) and d(embs[A], embs[C]).
    distAB = (embs[:, 0] - embs[:, 1]).pow(2).sum(-1).sqrt()
    distAC = (embs[:, 0] - embs[:, 2]).pow(2).sum(-1).sqrt()
    # The resulting tensor should be (N x 2 x num_cc_frames)

    # Sum over the number of frames in the trajectory. Negate.
    sumDistAB = -distAB.sum(-1).unsqueeze(1)
    sumDistAC = -distAC.sum(-1).unsqueeze(1)

    # Stack the Distances
    logits = torch.hstack((sumDistAB, sumDistAC))
    # The resulting tensors should be (N x 2)

    # Create the labels tensor.
    # Index CE Loss -- Always prefer d(embs[A], embs[B]) to d(embs[A], embs[C]). Data is structured this way.
    label_tensor = torch.zeros(logits.shape[0], dtype=torch.long, device=self._device)

    # Here, the size of inputs to CE loss needs to be the exact same.
    assert logits.shape[0] == label_tensor.shape[0]

    return F.cross_entropy(
        logits,
        label_tensor
    )
