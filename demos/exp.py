import gym
import xmagical
from gym.wrappers import Monitor
import os
from PIL import Image
xmagical.register_envs()
#print(xmagical.ALL_REGISTERED_ENVS)

# Create a demo variant for the SweepToTop task with a gripper agent.
#envname = 'SweepToTop-Gripper-Pixels-Allo-Demo-v0'
envname = 'SweepToTop-Longstick-Pixels-Allo-Demo-v0'
#envname = 'SweepToTop-Mediumstick-Pixels-Allo-Demo-v0'
#envname = 'SweepToTop-Shortstick-Pixels-Allo-Demo-v0'
env = gym.make(envname)
videopath = 'longstick/random/4/'
if not os.path.exists(videopath):
    os.makedirs(videopath)
obs = env.reset()
done = False
frame_count = 0
while not done:
    img = env.render(mode='rgb_array')
    frame_path = os.path.join(videopath, f"{frame_count}.png")
    img = Image.fromarray(img)
    img.save(frame_path)
    action = env.action_space.sample()
    next_obs, reward, done, info = env.step(action)
    env.render(mode='human')
    frame_count += 1
env.close()