import gym
import xmagical
from gym.wrappers import Monitor
import os
from PIL import Image
from typing import List
from pyglet.window import Window
from pyglet.window import key
import keyboard
import numpy as np
import time
from pynput import keyboard
import pyglet.window
xmagical.register_envs()

UP_DOWN_MAG = 0.5
ANGLE_MAG = np.radians(1.5)
OPEN_CLOSE_MAG = np.pi / 8
def get_action() -> List[float]:
    action = [0.0, 0.0, 0.0]
    _keys = key.KeyStateHandler()
    if _keys[key.UP] and not _keys[key.DOWN]:
        action[0] = +UP_DOWN_MAG
    elif (
            _keys[key.DOWN] and not _keys[key.UP]
    ):
        action[0] = -UP_DOWN_MAG
    if (
            _keys[key.LEFT]
            and not _keys[key.RIGHT]
    ):
        action[1] = ANGLE_MAG
    elif (
            _keys[key.RIGHT]
            and not _keys[key.LEFT]
    ):
        action[1] = -ANGLE_MAG
    if _keys[key.SPACE]:
        action[2] = OPEN_CLOSE_MAG

    return action[: 2]
# def getaction() -> List[float]:
#     action = [0.0, 0.0, 0.0]
#
#     if keyboard.is_pressed('up') and not keyboard.is_pressed('down'):
#         action[0] = 1.0
#     elif keyboard.is_pressed('down') and not keyboard.is_pressed('up'):
#         action[0] = -1.0
#
#     if keyboard.is_pressed('left') and not keyboard.is_pressed('right'):
#         action[1] = 1.0
#     elif keyboard.is_pressed('right') and not keyboard.is_pressed('left'):
#         action[1] = -1.0
#
#     if keyboard.is_pressed('space'):
#         action[2] = 1.0
#
#     return action[:2]

def on_press(key):
    try:
        if key == keyboard.Key.up:
            action[0] = 1.0
        elif key == keyboard.Key.down:
            action[0] = -1.0
        elif key == keyboard.Key.left:
            action[1] = 1.0
        elif key == keyboard.Key.right:
            action[1] = -1.0
        elif key == keyboard.Key.space:
            action[2] = 1.0
    except AttributeError:
        pass

def on_release(key):
    if key == keyboard.Key.up:
        action[0] = 0.0
    elif key == keyboard.Key.down:
        action[0] = 0.0
    elif key == keyboard.Key.left:
        action[1] = 0.0
    elif key == keyboard.Key.right:
        action[1] = 0.0
    elif key == keyboard.Key.space:
        action[2] = 0.0

def getaction() -> List[float]:
    return action[:3]
action = [0.0, 0.0, 0.0]
envname = 'SweepToTop-Gripper-Pixels-Allo-Demo-v0'
#envname = 'SweepToTop-Longstick-Pixels-Allo-Demo-v0'
#envname = 'SweepToTop-Mediumstick-Pixels-Allo-Demo-v0'
#envname = 'SweepToTop-Shortstick-Pixels-Allo-Demo-v0'
env = gym.make(envname)
obs = env.reset()
done = False
i = 0
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
videopath = 'gripper/ambiguous/4'
if not os.path.exists(videopath):
    os.makedirs(videopath)
frame_count = 0
while not done:
    img = env.render(mode='rgb_array')
    frame_path = os.path.join(videopath, f"{frame_count}.png")
    img = Image.fromarray(img)
    img.save(frame_path)
    action = getaction() #get_action() #env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    time.sleep(0.1)
    env.render(mode='human')
    frame_count += 1
listener.stop()
env.close()