from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch
import numpy as np

class PseudoCountModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, env, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.env = env
        print(self.env.observation_space.high)
        print(self.env.observation_space.low)
        self.histogram = np.zeros(self.env.observation_space.high.astype(int), self.env.observation_space.low.astype(int))
        self.n = 0

        # Because the Pointmass library has 2D floating point obs space
        # Discretize the counting
        # self.observation_space = gym.spaces.Box(
        # low=np.array([0,0]),
        # high=np.array([self._height, self._width]),
        # dtype=np.float32)

    def forward(self, ob_no):
        return ptu.from_numpy(self.forward_np(ptu.to_numpy(ob_no)))

    def forward_np(self, ob_no):
        obs_count = self.histogram[self.discretize_obs(ob_no)] + 1
        self.n += len(obs_count)
        # UCB reward bonus
        return np.sqrt(2* np.log(np.tile(self.n, len(obs_count))) / obs_count)

    def update(self, ob_no):
        self.histogram[self.discretize_obs(ob_no)] += 1
        return 0

    def discretize_obs(self, ob_no):
        print(ob_no)
        return np.floor(ob_no).astype(int)