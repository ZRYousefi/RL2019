import gym
import numpy as np
from matplotlib import pyplot as plt
import seaborn

def epsilonGreedy(Q, epsilon, num_actions, state):
    Action_probabilities = np.ones(num_actions,
                                   dtype=float) * epsilon / num_actions
    best_action = np.argmax(Q[state])
    Action_probabilities[best_action] += (1.0 - epsilon)
    return Action_probabilities


np.random.seed(123)

env = gym.make('CartPole-v0')
env.seed(321)

episodes = 10000
test_episodes = 10
num_of_actions = 2

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# For LunarLander, use the following values:
#         [  x     y  xdot ydot theta  thetadot cl  cr
# s_min = [ -1.2  -0.3  -2.4  -2  -6.28  -8       0   0 ]
# s_max = [  1.2   1.2   2.4   2   6.28   8       1   1 ]

# Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = np.int((episodes * target_eps) / (1-target_eps))  # TODO: Set the correct value.
initial_q = 0  # T3: Set to 50

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
v_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)

def get_discrete_state(state):
    discrete_state = np.zeros(len(state))
    discrete_state[0] = np.argmin(np.abs(x_grid - state[0]))
    discrete_state[1] = np.argmin(np.abs(v_grid - state[1]))
    discrete_state[2] = np.argmin(np.abs(th_grid - state[2]))
    discrete_state[3] = np.argmin(np.abs(av_grid - state[3]))
    return tuple(discrete_state.astype(np.int))

q_grid = np.zeros((discr, discr, discr, discr, num_of_actions)) + initial_q

# Training loop
ep_lengths, epl_avg = [], []
for ep in range(episodes+test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    epsilon = a / (a + ep) # 0.2    # T1: GLIE/constant, T3: Set to 0

    while not done:
        discrete_state = get_discrete_state(state)
        # TODO: IMPLEMENT HERE EPSILON-GREEDY
        #action = int(np.random.rand()*2)
        #action = np.argmax(q_grid[discrete_state])

        action_probabilities = epsilonGreedy(q_grid, epsilon, num_of_actions, discrete_state)

        # choose action according to
        # the probability distribution

        # if np.random.random() > epsilon:
        #     action = np.argmax(q_grid[discrete_state])
        # else:
        #     action = np.random.randint(0, num_of_actions)
        action = np.random.choice(np.arange(
            len(action_probabilities)),
            p=action_probabilities)

        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if not test:
            # TODO: ADD HERE YOUR Q_VALUE FUNCTION UPDATE
            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_grid[new_discrete_state])
            # Current Q value (for current state and performed action)
            current_q = q_grid[discrete_state + (action,  )]
            # equation for a new Q value for current state and action
            new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
            # Update Q table with new Q value
            q_grid[discrete_state + (action, )] = new_q

            if (done == True):
                current_q = q_grid[discrete_state + (action,)]
                new_q = (1 - alpha) * current_q + alpha * (gamma * current_q)
                q_grid[discrete_state + (action,)] = new_q

            pass
        else:
            env.render()
        state = new_state
        steps += 1
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))


# Calculate the value function
values = np.zeros(q_grid.shape[:-1])  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
values = np.sum(q_grid, axis=-1)
#(1 - target_eps) * np.max(q_grid, axis=4)  + (target_eps/num_of_actions) * np.sum(q_grid, axis=4)


# Plot the heatmap
# TODO: Plot the heatmap here using Seaborn or Matplotlib
heat = np.zeros([16, 16])
for i in range(16):
    for j in range(16):
        heat += values[:, i, :, j]
heat = heat / (16*16)

plt.imshow(heat)
plt.show()



# Save the Q-value array
np.save("q_values_eps_decreasing.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY
np.save("value_func_eps_decreasing.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY



seaborn.heatmap(heat)

# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()


"======================  Land Lunar  ======================"

import gym
import numpy as np
from matplotlib import pyplot as plt


def epsilonGreedy(Q, epsilon, num_actions, state):
    Action_probabilities = np.ones(num_actions,
                                   dtype=float) * epsilon / num_actions
    best_action = np.argmax(Q[state])
    Action_probabilities[best_action] += (1.0 - epsilon)
    return Action_probabilities


np.random.seed(123)

env = gym.make('LunarLander-v2')
env.seed(321)

episodes = 20000
test_episodes = 10
num_of_actions = env.action_space.n

# Reasonable values for Land Lunar discretization
discr = 16
bnr = 2

# For LunarLander, use the following values:
#         [  x     y  xdot ydot theta  thetadot cl  cr
s_min = [ -1.2, -0.3, -2.4,  -2,  -6.28,  -8,   0,   0 ]
s_max = [  1.2,  1.2,  2.4,   2,   6.28,   8,   1,   1 ]

def create_discrete_grid(s_min, s_max, discr, bnr):
    state_grid = []
    for i in range(len(s_min) - 2):
        state_grid.append( [np.linspace(s_min[i], s_max[i], discr)])
    state_grid.append([np.linspace(s_min[6], s_max[6], bnr)])
    state_grid.append([np.linspace(s_min[7], s_max[7], bnr)])
    return state_grid


    # Parameters
gamma = 0.98
alpha = 0.1
target_eps = 0.1
a = np.int((episodes * target_eps) / (1-target_eps))  # TODO: Set the correct value.
initial_q = 0  # T3: Set to 50

state_grid = create_discrete_grid(s_min, s_max, discr, bnr)
def get_discrete_state(state):
    discrete_state = np.zeros(len(state))
    for i in range(len(state) - 2):
        discrete_state[i] = np.argmin(np.abs(state_grid[i] - state[i]))
    discrete_state[6] = state[6]
    discrete_state[7] = state[7]
    return tuple(discrete_state.astype(np.int))

q_grid = np.zeros((discr, discr, discr, discr, discr, discr,
                   bnr, bnr, num_of_actions)) + initial_q


# Training loop
ep_lengths, epl_avg = [], []
for ep in range(episodes+test_episodes):
    test = ep > episodes
    state, done, steps = env.reset(), False, 0
    epsilon = a / (a + ep) # 0.2    # T1: GLIE/constant, T3: Set to 0

    while not done:
        discrete_state = get_discrete_state(state)
        # TODO: IMPLEMENT HERE EPSILON-GREEDY
        action_probabilities = epsilonGreedy(q_grid, epsilon, num_of_actions, discrete_state)

        # choose action according to
        # the probability distribution
        action = np.random.choice(np.arange(
            len(action_probabilities)),
            p=action_probabilities)

        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if not test:
            # TODO: ADD HERE YOUR Q_VALUE FUNCTION UPDATE
            max_future_q = np.max(q_grid[new_discrete_state])
            current_q = q_grid[discrete_state + (action,  )]
            new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_future_q)
            q_grid[discrete_state + (action, )] = new_q

            if (done == True):
                current_q = q_grid[discrete_state + (action,)]
                new_q = (1 - alpha) * current_q + alpha * (gamma * current_q)
                q_grid[discrete_state + (action,)] = new_q

            pass
        else:
            env.render()
        state = new_state
        steps += 1
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[max(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}".format(ep, np.mean(ep_lengths[max(0, ep-200):])))


# Calculate the value function
values = np.zeros(q_grid.shape[:-1])  # TODO: COMPUTE THE VALUE FUNCTION FROM THE Q-GRID
values = np.sum(q_grid, axis=-1)
#(1 - target_eps) * np.max(q_grid, axis=4)  + (target_eps/num_of_actions) * np.sum(q_grid, axis=4)

# Save the Q-value array
np.save("q_values.npy", q_grid)  # TODO: SUBMIT THIS Q_VALUES.NPY ARRAY
# Save the value_function array
np.save("value_func.npy", values)  # TODO: SUBMIT THIS VALUE_FUNC.NPY ARRAY


# Plot the heatmap
# TODO: Plot the heatmap here using Seaborn or Matplotlib
heat = np.zeros([16, 16])
for i in range(16):
    for j in range(16):
        heat += values[:, i, :, j]
heat = heat / (16*16)

plt.imshow(heat)
plt.show()


# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()

