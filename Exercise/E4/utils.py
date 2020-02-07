import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import random


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def plot_rewards(rewards):
    plt.figure(2)
    plt.clf()
    rewards_t = torch.tensor(rewards, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative reward')
    plt.grid(True)
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


class DQN(nn.Module):
    def __init__(self, state_space_dim, action_space_dim, hidden=32):
        super(DQN, self).__init__()
        self.hidden = hidden
        self.fc1 = nn.Linear(state_space_dim, hidden)
        self.fc2 = nn.Linear(hidden, action_space_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x




def epsilonGreedy(estimator, epsilon, num_actions):

    """
    :param estimator: An estimator that returns q values for a given state
    :param epsilon: The probability to select a random action . float between 0 and 1.
    :param num_actions: Number of actions in the environment.
    :return:
    A function that takes the states as an argument and returns the probabilities
    for each action in the form of a numpy array of length num_actions """

    def policy_fn(state):
        Action_probabilities = np.ones(num_actions,
                                       dtype=float) * epsilon / num_actions
        q_values = estimator.predict(state)
        best_action = np.argmax(q_values)
        Action_probabilities[best_action] += (1.0 - epsilon)
        return Action_probabilities

    return policy_fn