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

"""Teleop the agent and visualize the learned reward."""
import numpy as np
from absl import app
from absl import flags
from configs.constants import EMBODIMENTS
from configs.constants import XMAGICAL_EMBODIMENT_TO_ENV_NAME
from ml_collections import config_flags
import utils
from xmagical.utils import KeyboardEnvInteractor
import matplotlib.pyplot as plt
import json

FLAGS = flags.FLAGS

flags.DEFINE_enum("embodiment", "mediumstick", EMBODIMENTS,
                  "The agent embodiment.")
flags.DEFINE_boolean(
    "exit_on_done", True,
    "By default, env will terminate if done is True. Set to False to interact "
    "for as long as you want and press esc key to exit.")

config_flags.DEFINE_config_file(
    "config",
    "base_configs/rl.py",
    "File path to the training hyperparameter configuration.",
)
def main(_):
    env_name = XMAGICAL_EMBODIMENT_TO_ENV_NAME[FLAGS.embodiment]
    env = utils.make_env(env_name, seed=3)

    # ACTIONS_FOLDER = "/home/connor/Documents/Xpref/trajectories_num_blocks/valid/mediumstick/25/actions.json"
    # ACTIONS_FOLDER = "/home/connor/Documents/Xpref/trajectories_num_blocks/valid/mediumstick/932/actions.json"
    ACTIONS_FOLDER = "/home/connor/Documents/Xpref/trajectories_num_blocks/valid/mediumstick/1763/actions.json"
    with open(ACTIONS_FOLDER, "r") as f:
        actions = json.load(f)

    # Reward learning wrapper.
    if FLAGS.config.reward_wrapper.type is not None:
        env = utils.wrap_learned_reward(env, FLAGS.config)

    viewer = KeyboardEnvInteractor(action_dim=env.action_space.shape[0])

    LIVE_REWARD = False
    # MAX_STEPS = len(actions)
    MAX_STEPS = 100

    env.reset()
    obs = env.render("rgb_array")
    viewer.imshow(obs)

    i = [0]
    rews = []
    ONLY_UPDATE_PLOT_ON_STEP = False


    if LIVE_REWARD:
        plt.ion()

    def step(action):
        # if np.linalg.norm(np.array(action)) < 0.1:
        #     return
        if i[0] < MAX_STEPS:
            action = actions[i[0]]
        else:
            action = [0.0, 0.0]

        obs, rew, done, info = env.step(action)
        if not ONLY_UPDATE_PLOT_ON_STEP or sum(action) != 0.0:
            rews.append(rew)
        if LIVE_REWARD:
            plt.clf()
            plt.plot(rews)
            plt.pause(0.01)
            plt.show()
        if obs.ndim != 3:
            obs = env.render("rgb_array")
        if done:
            print(f"Done, score {info['eval_score']:.2f}/1.00")
            print("Episode metrics: ")
            for k, v in info["episode"].items():
                print(f"\t{k}: {v}")
            if FLAGS.exit_on_done:
                return
        if i[0] > MAX_STEPS:
            return
        i[0] += 1
        return obs


    viewer.run_loop(step)
    plt.ioff()

    np.savetxt("reward_out.txt", np.array(rews))

    # utils.plot_reward(rews)


if __name__ == "__main__":
    app.run(main)
