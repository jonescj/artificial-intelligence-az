# -*- coding: utf-8 -*-
"""Copy of Deep Q-Learning for Lunar Landing - Partial Code

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Hw80E_UuatP-8n4JpqdwqpMQaHksXHGI

# Deep Q-Learning for Lunar Landing

## Part 0 - Installing the required packages and importing the libraries

### Installing Gymnasium
"""

!pip install gymnasium
!pip install "gymnasium[atari, accept-rom-license]"
!apt-get install -y swig
!pip install gymnasium[box2d]

"""### Importing the libraries"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque, namedtuple

"""## Part 1 - Building the AI

### Creating the architecture of the Neural Network

## Part 2 - Training the AI

### Setting up the environment

### Initializing the hyperparameters

### Implementing Experience Replay

### Implementing the DQN class

### Initializing the DQN agent

### Training the DQN agent

## Part 3 - Visualizing the results
"""

import glob
import io
import base64
import imageio
from IPython.display import HTML, display
from gym.wrappers.monitoring.video_recorder import VideoRecorder

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action.item())
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

show_video_of_model(agent, 'LunarLander-v2')

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