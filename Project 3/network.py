import torch
import torch.nn as nn
import numpy as np

# pylint: disable=E1101

def hidden_init(layer):
    """Calculates the initialization parameters for the hidden layers
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model.
    
    Based on the code from ddpg-pendulum.
    """

    def __init__(self, in_size, fc_units, out_size):
        """Initialize parameters and build model.
        Params
        ======
            in_size (int): Dimension of input tensor
            fc_units (list int): Number of nodes in the hidden layers
            out_size (int): Dimension of output tensor

        """
        super(Actor, self).__init__()
        units = [in_size] + fc_units + [out_size]
        layers = []
        for i in range(len(units)-2):
            layer = nn.Linear(units[i], units[i+1])
            layer.weight.data.uniform_(*hidden_init(layer))
            layers.append(layer)
            layers.append(nn.ReLU())
        # last layer: different activation and initialization
        layer = nn.Linear(units[-2], units[-1])
        layer.weight.data.uniform_(-3e-3, 3e-3)
        layers.append(layer)
        layers.append(nn.Tanh())
        self.net = nn.ModuleList(layers)

        
    def forward(self, data_in):
        """Build an actor (policy) network that maps states -> actions."""
        x = data_in
        for layer in self.net:
            x = layer(x)
        return x


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, fc_units, num_agents):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc_units (list of 2 list of int): Number of nodes in the hidden layers:
                    there should be 2 lists, one for the nodes in the layer stack
                    that processes the state input and one stack that processes
                    the concatenated result of first stack and the actions
            num_agents (int): the number of agents being critiqued
        """
        super(Critic, self).__init__()
         
        # state layer stack
        state_units = [state_size] + fc_units[0]
        state_layers = []
        for i in range(len(state_units)-1):
            layer = nn.Linear(state_units[i], state_units[i+1])
            layer.weight.data.uniform_(*hidden_init(layer))
            state_layers.append(layer)
            state_layers.append(nn.ReLU())
        self.state_layers = nn.ModuleList(state_layers)
            
        # action layer stack
        action_units = [action_size + fc_units[0][-1]] + fc_units[1] + [num_agents]
        action_layers = []
        for i in range(len(action_units)-2):
            layer = nn.Linear(action_units[i], action_units[i+1])
            layer.weight.data.uniform_(*hidden_init(layer))
            action_layers.append(layer)
            action_layers.append(nn.ReLU())
        # last layer: no activation, different initialization
        layer = nn.Linear(action_units[-2], action_units[-1])
        layer.weight.data.uniform_(-3e-3, 3e-3)
        action_layers.append(layer)
        self.action_layers = nn.ModuleList(action_layers)


    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = state
        for layer in self.state_layers:
            xs = layer(xs)
        x = torch.cat((xs, action), dim=1)
        for layer in self.action_layers:
            x = layer(x)
        return x


class DoubleNN(nn.Module):
    """A class that implements a local / target double network.
    Created for convenience to simplify saving the parameters and
    performing soft copy from local to target.
    """

    def __init__(self, localNN, targetNN):
        super(DoubleNN, self).__init__()

        self.local = localNN
        self.target = targetNN
        # copy the parameters from local to target
        self.soft_update(1)


    def soft_update(self, tau):
        """Performs a fost copy from local to target NN
        """
        for target_param, local_param in zip(self.target.parameters(), 
                                             self.local.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def forward(self):
        """By default the forward propagation will call the local NN
        forward method
        """
        self.local.forward()


    def target_forward(self):
        """For convenience
        """
        self.target.forward()


class MADDPGNet(nn.Module):
    """A class that implements an arbitrary number of Actors (agents) and
    one critic.
    """

    def __init__(self, num_agents, obs_size, action_size,
                 actor_fc_units, critic_fc_units):
        super(MADDPGNet, self).__init__()
        actors = []
        for _ in range(num_agents):
            actor = DoubleNN(localNN=Actor(in_size=obs_size, 
                                           fc_units=actor_fc_units,
                                           out_size=action_size),
                             targetNN=Actor(in_size=obs_size,
                                            fc_units=actor_fc_units,
                                            out_size=action_size))
            actors.append(actor)
        self.actors = nn.ModuleList(actors)

        self.critic = DoubleNN(localNN=Critic(state_size=obs_size*num_agents,
                                              action_size=action_size*num_agents,
                                              fc_units=critic_fc_units,
                                              num_agents=num_agents),
                               targetNN=Critic(state_size=obs_size*num_agents,
                                               action_size=action_size*num_agents,
                                               fc_units=critic_fc_units,
                                               num_agents=num_agents))


    def forward(self):
        # we actually cannot use the net directly, we will need to use
        # the individual components
        pass

            
    def actor_soft_update(self, tau):
        """Convenience method. Updates the actors' target net weight"""
        for actor in self.actors:
            actor.soft_update(tau)

        
    def critic_soft_update(self, tau):
        """Convenience method. Updates the critic's target net weight"""
        self.critic.soft_update(tau)


    def actors_local(self):
        """Returns the local nets for actors
        """
        return [actor.local for actor in self.actors]


    def actors_target(self):
        """Returns the target nets for actors
        """
        return [actor.target for actor in self.actors]


    def soft_update(self, tau):
        """Soft update all model parameters.
         """
        self.actor_soft_update(tau)
        self.critic_soft_update(tau)

# pylint: enable=E1101