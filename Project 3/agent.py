import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from network import MADDPGNet
from OUNoise import OUNoise

# pylint: disable=E1101

class MADDPGAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, num_agents,
                 state_size, action_size,
                 actor_units, critic_units,
                 memory,
                 batch_size,
                 device='cpu',
                 lr_actor=1e-4, lr_critic=1e-4,
                 weight_decay=0,
                 gamma=0.99,
                 tau=1e-3,
                 update_every = 10):
        """Initialize an Agent object.
        
        Params
        ======
            num_agents (int): number of agents
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            actor_units (list int): configuration of layers for Actor nets
            critic_units (list list int): configuration of layers for Critic nets
            memory (ReplayBuffer): pre-created and loaded memory buffer to use
            batch_size (int): the size of the sampling from the replay buffer
            device (str): the hardware where the computation will happen
            lr_actor (float): learning rate for actor network
            lr_critic (float): learning rate for critic network
            weight_decay (float): weight decay for the critic optimizer
            gamma (float): discount factor for return
            tau (float): factor for soft update of target networks
            update_every (int): after how many steps we perform a training
        """
        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        self.actor_units = actor_units
        self.critic_units = critic_units
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.steps = 0
        self.update_every = update_every   
              
        # setup the nets: one for each agent
        self.net = MADDPGNet(num_agents=num_agents,
                             obs_size=state_size,
                             action_size=action_size,
                             actor_fc_units=actor_units,
                             critic_fc_units=critic_units).to(device)
        
        actor_for_opt = torch.nn.ModuleList([actor.local for actor in self.net.actors])
        self.actor_optimizer = optim.Adam(actor_for_opt.parameters(), 
                                          lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.net.critic.local.parameters(),
                                           lr=lr_critic, 
                                           weight_decay=weight_decay)

        # Noise process
        self.noise = OUNoise(action_size)

        # Replay memory
        self.memory = memory
        self.batch_size = batch_size
        self.device = device

        
    def step(self, obs, actions, rewards, next_obs, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.push(obs, actions, rewards, next_obs, dones)

        # Learn, if enough samples are available in memory
        self.steps += 1
        if len(self.memory) > self.batch_size and self.steps % self.update_every == 0:
            self.learn()

            
    def act(self, obs, eps, add_noise=True):
        """Returns actions for given state as per current policy
        for each agent.
        eps (float): a decay factor for application of noise
        """
        actions = []
        for agent in range(self.num_agents):
            ag_obs = torch.from_numpy(obs[agent]).float().to(self.device)
            curr_actor = self.net.actors_local()[agent]
            curr_actor.eval()
            with torch.no_grad():
                action = curr_actor(ag_obs).cpu().data.numpy()
            curr_actor.train()
            if add_noise:
                action += eps * self.noise()
            action = np.clip(action, -1, 1)
            actions.append(action)
        return np.array(actions)


    def reset(self):
        self.noise.reset()

        
    def learn(self):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        """
        # draw samples
        experiences = self.memory.sample(self.batch_size, self.device)
        # unpack experience
        obs, actions, rewards, next_obs, dones = experiences
        # remember these are provided: samples x agents x data dimension
        n_samples = obs.size()[0]

        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_obs_t = torch.transpose(next_obs, 0, 1)
        actions_next = [self.net.actors_target()[agent](next_obs_t[agent]) \
                    for agent in range(self.num_agents)]
        actions_next = torch.stack(actions_next)
        actions_next = torch.transpose(actions_next, 0, 1)
        actions_next= torch.reshape(actions_next, (n_samples, -1))

        next_obs_2d = torch.reshape(next_obs, (n_samples, -1))
 
        Q_targets_next = self.net.critic.target(next_obs_2d, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        obs_2d = torch.reshape(obs, (n_samples, -1))
        actions_2d = torch.reshape(actions, (n_samples, -1))
        Q_expected = self.net.critic.local(obs_2d, actions_2d)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.critic.local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        obs_t = torch.transpose(obs, 0, 1)
        actions_pred = [self.net.actors_local()[agent](obs_t[agent]) \
                    for agent in range(self.num_agents)]
        actions_pred = torch.stack(actions_pred)
        actions_pred = torch.transpose(actions_pred, 0, 1)
        actions_pred= torch.reshape(actions_pred, (n_samples, -1))
        
        actor_loss = - self.net.critic.local(obs_2d, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.net.soft_update(self.tau)               

            
    def save(self, filename):
        """Save the model parameters.
        
        Params
        ======
            filename (str): file name including path
            
            The method will save the weights of all networks in the
            specified directory
        """
        torch.save(self.net.state_dict(), filename)
        
        
    def load(self, filename):
        """
        Loads the model parameters.
        All agents will be initialized to the same paramters as read from 
        the file.
        
        Params
        ======
            filename (str): file name including path
            
            The method will save the weights of all networks in the
            specified directory
        """
        self.net.load_state_dict(torch.load(filename))

# pylint: enable=E1101