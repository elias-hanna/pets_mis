from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class BallInCupEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    pass
