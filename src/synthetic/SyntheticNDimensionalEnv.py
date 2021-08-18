import math
import numpy as np
from gym import Env, spaces
import sys
import random

from src.synthetic.SyntheticNDimensional import SyntheticGaussianNumpy


class SyntheticGaussianEnv(Env):

    def __init__(self,
                 synthetic_gaussian_params,
                 random=np.random.RandomState(0),
                 noise_std=0
                 ):
        self.synthetic_gaussian_params = synthetic_gaussian_params
        self.noise_std = noise_std
        self.random = random
        self.snd = None
        self.reset()
        min_x = self.snd.min_x
        max_x = self.snd.max_x
        dim = self.snd.dim
        if "random" not in self.synthetic_gaussian_params or self.synthetic_gaussian_params["random"] is None:
            self.synthetic_gaussian_params["random"] = self.random
        self.action_space = spaces.Box(low=np.array([min_x] * dim), high=np.array([max_x] * dim), dtype=np.float16)

    def render(self, mode='human', close=False):
        pass

    def reset(self):
        self.snd = SyntheticGaussianNumpy(**self.synthetic_gaussian_params)

    def step(self, action=None, debug=False):  # either geometric or poisson (geometric for now)
        if len(action.shape)==1:
            action = action.reshape(1, -1)
        r_average = self.snd.compute_function_on_batch(action)
        r = self.random.normal(r_average, scale=self.noise_std)
        return None, r, None, None
