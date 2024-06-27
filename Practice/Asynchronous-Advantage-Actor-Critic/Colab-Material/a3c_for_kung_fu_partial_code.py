# -*- coding: utf-8 -*-
"""A3C for Kung Fu - Partial Code

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1x10CV4c957vpJh1ruNsoQwKCJlkiTp1s

# A3C for Kung Fu

## Part 0 - Installing the required packages and importing the libraries

### Installing Gymnasium
"""

!pip install gymnasium
!pip install "gymnasium[atari, accept-rom-license]"
!apt-get install -y swig
!pip install gymnasium[box2d]

"""### Importing the libraries"""

import cv2
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributions as distributions
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

"""## Part 1 - Building the AI

### Creating the architecture of the Neural Network

## Part 2 - Training the AI

### Setting up the environment
"""

class PreprocessAtari(ObservationWrapper):

  def __init__(self, env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4):
    super(PreprocessAtari, self).__init__(env)
    self.img_size = (height, width)
    self.crop = crop
    self.dim_order = dim_order
    self.color = color
    self.frame_stack = n_frames
    n_channels = 3 * n_frames if color else n_frames
    obs_shape = {'tensorflow': (height, width, n_channels), 'pytorch': (n_channels, height, width)}[dim_order]
    self.observation_space = Box(0.0, 1.0, obs_shape)
    self.frames = np.zeros(obs_shape, dtype = np.float32)

  def reset(self):
    self.frames = np.zeros_like(self.frames)
    obs, info = self.env.reset()
    self.update_buffer(obs)
    return self.frames, info

  def observation(self, img):
    img = self.crop(img)
    img = cv2.resize(img, self.img_size)
    if not self.color:
      if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32') / 255.
    if self.color:
      self.frames = np.roll(self.frames, shift = -3, axis = 0)
    else:
      self.frames = np.roll(self.frames, shift = -1, axis = 0)
    if self.color:
      self.frames[-3:] = img
    else:
      self.frames[-1] = img
    return self.frames

  def update_buffer(self, obs):
    self.frames = self.observation(obs)

def make_env():
  env = gym.make("KungFuMasterDeterministic-v0", render_mode = 'rgb_array')
  env = PreprocessAtari(env, height = 42, width = 42, crop = lambda img: img, dim_order = 'pytorch', color = False, n_frames = 4)
  return env

env = make_env()

state_shape = env.observation_space.shape
number_actions = env.action_space.n
print("Observation shape:", state_shape)
print("Number actions:", number_actions)
print("Action names:", env.env.env.get_action_meanings())

"""### Initializing the hyperparameters

### Implementing the A3C class

### Initializing the A3C agent

### Evaluating our A3C agent on a single episode

### Testing multiple agents on multiple environments at the same time

### Training the A3C agent

## Part 3 - Visualizing the results
"""

import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(agent, env):
  state, _ = env.reset()
  done = False
  frames = []
  while not done:
    frame = env.render()
    frames.append(frame)
    action = agent.act(state)
    state, reward, done, _, _ = env.step(action[0])
  env.close()
  imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, env)

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()