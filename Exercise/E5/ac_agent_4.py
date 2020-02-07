import torch
import torch.nn.functional as F
from torch.distributions import Normal
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
        self.sigma = torch.nn.Parameter(10*torch.ones(1))
        # critic's layer
        self.fc2_value = torch.nn.Linear(self.hidden, 1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        mu = self.fc2_mean(x)
        sigma = F.softplus(self.sigma)
        action_distribution = Normal(mu, sigma)
        # TODO: Add a layer for state value calculation (T3)
        # critic: evaluates being in the state s_t
        state_values = self.fc2_value(x)
        return action_distribution, state_values

class Agent(object):
    def __init__(self, policy):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.delta = []

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0) \
            .to(self.train_device).squeeze(-1)
        delta = torch.stack(self.delta, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards, self.delta = [], [], [], []

        # TODO: Compute the optimization term (T1, T3)
        # calculate actor (policy) loss
        actor_losses = (-action_probs * delta).sum()
        # calculate critic (value) loss
        critic_losses = delta.pow(2).sum()

        self.optimizer.zero_grad()
        loss = actor_losses + critic_losses
        loss.backward()
        self.optimizer.step()

    def each_timestep(self, observation, reward, done, critic_value):
        x = torch.from_numpy(observation).float().to(self.train_device)
        # TODO: Pass state x through the policy network (T1)
        _, critic_value_prime = self.policy(x.detach())
        # Calculate TD(0) error
        self.delta.append(reward + self.gamma * critic_value_prime * (1 - int(done)) - critic_value)


    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.train_device)

        # TODO: Pass state x through the policy network (T1)
        action_dist, state_value = self.policy(x)

        # TODO: Return mean if evaluation, else sample from the distribution
        # returned by the policy (T1)
        if evaluation:
            action = action_dist.mean()
        else:
            # sample an action using the distribution
            action = action_dist.sample()

        # TODO: Calculate the log probability of the action (T1)
        act_log_prob = action_dist.log_prob(action)

        # TODO: Return state value prediction, and/or save it somewhere (T3)
        return action, act_log_prob, state_value

    def store_outcome(self, observation, action_prob, reward):
        self.states.append(observation)
        self.action_probs.append(action_prob)
        self.rewards.append(torch.Tensor([reward]))