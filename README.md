# Representation Alignment from Human Feedback for Cross-Embodiment Reward Learning from Mixed-Quality Demonstrations
**Connor Mattson, Anurag Aribandi, and Daniel S. Brown**


[![python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-383/)

## Overview

We seek to learn reward functions from demonstrations given by multiple different robot models (mixed-embodiment) and with varying task success (mixed-quality), denoted "MQME" data. Our work explores several options for reward learning including preference learning, triplet learning, and TCC augmentation. Please refer to [our paper](https://sites.google.com/view/cross-irl-mqme/home) for more details. 

This repository serves as a foundation for exploring and extending our models, as well as reproducing the results of the paper.

This codebase is heavily authored by and reliant on the work published by [Zakka et al. "XIRL: Cross-embodiment Inverse Reinforcement Learning"](https://x-irl.github.io/). Their published code served as the foundation for this project. [Here is the original repository we forked from](https://github.com/google-research/google-research/tree/master/xirl).

For the latest updates, videos, and the open-access paper, [visit our project website](https://sites.google.com/view/cross-irl-mqme/home).

## Setup

We use Python 3.8 and [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for development. To create an environment and install dependencies, run the following steps:

```bash
# Clone and cd into xprefs.
git clone git@github.com:anurag-src/xpref.git

# Create and activate environment.
conda create -n xpref python=3.8
conda activate xpref

# Install dependencies.
pip install -r requirements.txt
```

## Datasets

**MQME Dataset**

You can download and unzip the demonstration dataset used in our work from [this Google Drive link](https://drive.google.com/file/d/1xvUv3LlrBzVJDhE3zk3W_e8mK9lv5ckU/view?usp=sharing).

**Goal Examples Dataset**

See [this Google Drive Link](https://drive.google.com/file/d/16m0FKz8SR6kMcZKy1oABEFe0HlPnjAOp/view?usp=sharing).

**X-MAGICAL (Optional, for baselines)**

Run the following bash script to download the demonstration dataset for the [X-MAGICAL benchmark](https://github.com/kevinzakka/x-magical):

```bash
bash scripts/download_xmagical_dataset.sh
```

The dataset will be located in `/tmp/xirl/datasets/xmagical`. You are free to modify the save destination, just make sure you update `config.data.root` in the pretraining config file &mdash; see `base_configs/pretrain.py`.

**Set your dataset directory**

In [base_configs/pretrain.py](base_configs/pretrain.py), update the value of "config.data.root" to match the file path where your dataset is. For the XMagical dataset, we recommend moving the data from the default tmp installation directory if you plan to benchmark or build upon our work and XIRL.

## Experiments

For all experiments, configure the "config.root_dir" variable in [base_configs/pretrain.py](base_configs/pretrain.py) to an experiment output directory (rollout videos, saved models, tensorboard logs, etc.)

### Reward Learning (7 Methods)
#### XIRL (Baseline)
1. In [base_configs/pretrain.py](base_configs/pretrain.py), set "config.data.root" to the path where the X-Magical dataset lies.
2. Run `python pretrain.py --config configs/xmagical/pretraining/tcc.py`
3. Run `python compute_goal_embedding.py --experiment_path [PATH_TO_PRETRAIN_MODEL] --withheld_goals False`

#### XIRL Mixed (Baseline)
1. In [base_configs/pretrain.py](base_configs/pretrain.py), set "config.data.root" to the path where the MQME dataset lies.
2. In [base_configs/xprefs.py](base_configs/xprefs.py), set "config.data.goal_examples" to the path where your goal examples dataset is.
3. Run `python pretrain.py --config configs/xmagical/pretraining/tcc.py`
4. Run `python compute_goal_embedding.py --experiment_path [PATH_TO_PRETRAIN_MODEL] --withheld_goals True`

#### Goal Classifier (Baseline)
1. In [base_configs/pretrain.py](base_configs/pretrain.py), set "config.data.root" to the path where the X-Magical dataset lies.
2. Run `python pretrain.py --config configs/xmagical/pretraining/classifier.py`

#### X-RLHF (Ours)
1. In [base_configs/xprefs.py](base_configs/xprefs.py), set "config.data.demonstrations_root" to the path where the MQME dataset lies.
2. In the same config file, set "config.irl.learning_type" to "RLHF"
3. Run `python xprefs_irl.py`

#### XPrefs (Ours)
1. In [base_configs/xprefs.py](base_configs/xprefs.py), set "config.data.demonstrations_root" to the path where the MQME dataset lies.
2. In the same config file, set "config.irl.learning_type" to "XPrefs"
3. In the same config file, set "config.data.goal_examples" to the path where your goal examples dataset is.
4. Run `python xprefs_irl.py`
5. Run `python compute_goal_embedding.py --experiment_path [PATH_TO_PRETRAIN_MODEL] --withheld_goals True`

#### XTriplets (Ours)
1. In [base_configs/pretrain.py](base_configs/pretrain.py), set "config.data.demonstrations_root" to the path where the MQME dataset lies.
2. In [base_configs/xprefs.py](base_configs/xprefs.py), set "config.data.goal_examples" to the path where your goal examples dataset is.
3. Run `python pretrain.py --config configs/xmagical/pretraining/triplets.py`
3. Run `python compute_goal_embedding.py --experiment_path [PATH_TO_PRETRAIN_MODEL] --withheld_goals True`

#### XIRL-Buckets (Ours)
1. In [base_configs/pretrain.py](base_configs/pretrain.py), set "config.data.demonstrations_root" to the path where the MQME dataset lies.
2. In [base_configs/xprefs.py](base_configs/xprefs.py), set "config.data.goal_examples" to the path where your goal examples dataset is.
3. Run `python xprefs/bucket_trainer`
4. Run `python compute_goal_embedding.py --experiment_path [PATH_TO_PRETRAIN_MODEL] --withheld_goals True`


### Reinforcement Learning

#### Ground Truth Reward
1. Run `python rl_xmagical_env_reward.py --embodiment mediumstick`

#### For all other learned methods, follow these instructions.
1. In [base_configs/rl.py](base_configs/rl.py), set the value of "config.reward_wrapper.type" to the appropriate value from the list below.
   1. `'distance_to_goal'` if reward learning was XIRL, XIRL Mixed, XTriplets, or XIRL-Buckets
   2. `'goal_classifier'` if reward learning was "Goal Classifier"
   3. `'reward_prediction_from_prefs'` if reward learning was XPrefs
   4. `'RLHF'` if reward learning was "X-RLHF".
2. In the same config file, set config.reward_wrapper.pretrained_path to the experiment directory where the reward learning model was saved.
3. Run `python rl_xmagical_learned_reward.py --pretrained_path [PATH_TO_PRETRAINING_DIR]`

## Code Navigation

Please refer to the [original repository](https://x-irl.github.io/) for up-to-date information on code navigation and structure. They are the engineers behind the great architecture and usability.
