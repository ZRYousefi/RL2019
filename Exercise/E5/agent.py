import torch
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from utils import discount_rewards


class Policy(torch.nn.Module):
	def __init__(self, state_space, action_space, std_mode='1'):
		super().__init__()
		"""
		std_mode is '1' for task1 and '2a' or '2b' for task 2
		"""
		self.state_space = state_space
		self.action_space = action_space
		self.hidden = 64
		self.fc1 = torch.nn.Linear(state_space, self.hidden)
		self.fc2_mean = torch.nn.Linear(self.hidden, action_space)
		self.sigma = 5 if '1' in std_mode else 10 # TODO: Implement accordingly (T1, T2)
		if std_mode == '2b':
			self.sigma = torch.nn.Parameter(10*torch.ones(1))
		self.init_weights()
		self.std_mode = std_mode
		self.c = 5e-4

	def init_weights(self):
		for m in self.modules():
			if type(m) is torch.nn.Linear:
				torch.nn.init.normal_(m.weight)
				torch.nn.init.zeros_(m.bias)

	def forward(self, x, episode_number):
		x = self.fc1(x)
		x = F.relu(x)
		mu = self.fc2_mean(x)
		if self.std_mode == '2a':
			sigma = self.sigma * np.exp(-self.c * episode_number)  # TODO: Is it a good idea to leave it like this?
		elif self.std_mode == '2b':
			sigma = F.softplus(self.sigma)
		else:
			sigma = self.sigma

		# TODO: Instantiate and return a normal distribution
		# with mean mu and std of sigma (T1)
		action_distribution = Normal(mu, sigma)
		return action_distribution
		# TODO: Add a layer for state value calculation (T3)


class Agent(object):
	def __init__(self, policy):
		self.train_device = "cpu"
		self.policy = policy.to(self.train_device)
		self.optimizer = torch.optim.RMSprop(policy.parameters(), lr=5e-3)
		self.gamma = 0.98
		self.states = []
		self.action_probs = []
		self.rewards = []

	def episode_finished(self, episode_number, baseline_mode=1):
		"""
		baseline_mode is 1 for Task1_a,
						 2 for Task1_b,
						 3 for Task1_c
		"""
		action_probs = torch.stack(self.action_probs, dim=0) \
				.to(self.train_device).squeeze(-1)
		rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
		self.states, self.action_probs, self.rewards = [], [], []

		# TODO: Compute discounted rewards (use the discount_rewards function)
		discounted_rewards = discount_rewards(rewards, self.gamma)
		if baseline_mode == 3:
			discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-10)
		# TODO: Compute critic loss and advantages (T3)

		# TODO: Compute the optimization term (T1, T3)
		b = 20 if baseline_mode == 2 else 0
		loss = (-action_probs * (discounted_rewards - b)).sum()

		# TODO: Compute the gradients of loss w.r.t. network parameters (T1)
		# TODO: Update network parameters using self.optimizer and zero gradients (T1)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()


	def get_action(self, observation, episode_number, evaluation=False):
		x = torch.from_numpy(observation).float().to(self.train_device)

		# TODO: Pass state x through the policy network (T1)
		action_dist = self.policy.forward(x, episode_number)

		# TODO: Return mean if evaluation, else sample from the distribution
		# returned by the policy (T1)
		if evaluation:
			action = action_dist.mean()
		else:
			action = action_dist.sample()

		# TODO: Calculate the log probability of the action (T1)
		act_log_prob = action_dist.log_prob(action)

		# TODO: Return state value prediction, and/or save it somewhere (T3)

		return action, act_log_prob

	def store_outcome(self, observation, action_prob, action_taken, reward):
		self.states.append(observation)
		self.action_probs.append(action_prob)
		self.rewards.append(torch.Tensor([reward]))
