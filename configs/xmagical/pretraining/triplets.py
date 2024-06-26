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

"""TCC config."""

from base_configs.pretrain import get_config as _get_config


def get_config():
  """TCC config."""

  config = _get_config()

  print("USING TRIPLET CONFIG")
  config.algorithm = "triplets"
  config.optim.train_max_iters = 4_000
  config.data.batch_size = 32
  config.frame_sampler.strategy = "uniform"
  config.frame_sampler.uniform_sampler.offset = 0

  config.frame_sampler.num_frames_per_sequence = 15
  config.frame_sampler.num_context_frames = 1
  config.frame_sampler.context_stride = 3

  # config.frame_sampler.num_frames_per_sequence = 40

  config.model.model_type = "resnet18_linear"
  config.model.embedding_size = 32
  config.model.normalize_embeddings = False
  config.model.learnable_temp = False

  # Learning rate.
  config.optim.lr = 1e-5

  config.data_augmentation.train_transforms = [
    "global_resize",
    "random_resized_crop",
    "color_jitter",
    "grayscale",
    "gaussian_blur",
    # "normalize",
  ]
  config.data_augmentation.eval_transforms = [
    "global_resize",
    # "normalize",
  ]

  ## Remove this when you need eval again
  config.eval.val_iters = 1
  config.eval.eval_frequency = 1500

  # config.loss.tcc.stochastic_matching = False
  # config.loss.tcc.loss_type = "regression_mse"
  # config.loss.tcc.similarity_type = "l2"
  # config.loss.tcc.softmax_temperature = 1.0

  return config
