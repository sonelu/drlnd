from collections import deque, namedtuple
import random
import pickle
import torch
import numpy as np


Experience = namedtuple("Experience", 
                        field_names=["obs", 
                                     "actions", 
                                     "rewards", 
                                     "next_obs", 
                                     "dones"])

# pylint: disable=E1101

class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.memory = deque(maxlen=self.size)


    def push(self,obs, actions, rewards, next_obs, dones):
        """push into the buffer"""
        e = Experience(obs, actions, rewards, next_obs, dones)
        self.memory.append(e)


    def sample(self, batchsize, device='cpu'):
        """sample from the buffer"""
        samples = random.sample(self.memory, batchsize)

        # get each item from Experience as a tensor
        obs = torch.from_numpy(np.stack([e.obs for e in samples \
                            if e is not None])).float().to(device)

        # repeat for actions
        actions = torch.from_numpy(np.stack([e.actions for e in samples \
                            if e is not None])).float().to(device)
        
        # rewards
        rewards = torch.from_numpy(np.stack([e.rewards for e in samples \
                            if e is not None])).float().to(device)

        # next observations
        next_obs = torch.from_numpy(np.stack([e.next_obs for e in samples \
                            if e is not None])).float().to(device)

        # dones
        dones = torch.from_numpy(np.stack([e.dones for e in samples \
                            if e is not None]).astype(np.uint8)).float().to(device)

        return (obs, actions, rewards, next_obs, dones)
 

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.memory, f)


    def load(self, filename):
        with open(filename, 'rb') as f:
            self.memory = pickle.load(f)


    def __len__(self):
        return len(self.memory)

# pylint: enable=E1101

