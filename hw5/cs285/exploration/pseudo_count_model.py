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
        self.histogram = np.zeros(self.env.observation_space.high.astype(int))
        self.n = 0

        # Because the Pointmass library has 2D floating point obs space
        # Discretize the counting
        # self.observation_space = gym.spaces.Box(
        # low=np.array([0,0]),
        # high=np.array([self._height, self._width]),
        # dtype=np.float32)
    def lookup_histogram(self, ob_no):
        discretized_obs = np.floor(ob_no).astype(int)
        return [self.histogram[(ob[0], ob[1])] for ob in discretized_obs]

    def forward(self, ob_no):
        return ptu.from_numpy(self.forward_np(ptu.to_numpy(ob_no)))

    def forward_np(self, ob_no):
        obs_count = np.array(self.lookup_histogram(ob_no)) +1
        self.n += len(obs_count)
        self.update(ob_no)
        ucb_bonus = np.sqrt(2* np.log(np.tile(self.n, len(obs_count))) / obs_count)
        # UCB reward bonus
        return ucb_bonus

    def update(self, ob_no):
        discretized_obs = np.floor(ob_no).astype(int)
        for ob in discretized_obs:
          self.histogram[(ob[0], ob[1])] += 1
        return 0