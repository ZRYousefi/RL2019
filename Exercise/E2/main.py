import numpy as np
from time import sleep
from sailing import SailingGridworld
import matplotlib.pyplot as plt


# Set up the environment
env = SailingGridworld(rock_penalty=-2)
value_est = np.zeros((env.w, env.h))
env.draw_values(value_est)


if __name__ == "__main__":

    "==================  TAsk 1 and 2  ================"
    # Reset the environment
    state = env.reset()

    # Compute state values and the policy
    # TODO: Compute the value function and policy (Tasks 1, 2 and 3)
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))

    n_actions = env.transitions.shape[2]
    gamma = 0.9
    Q = np.zeros((env.w, env.h, n_actions))
    for _ in range(100):
        for i in range(env.w):
            for j in range(env.h): #-1,-1,-1):
                for a in range(n_actions):
                    value_s_primes = 0
                    for s in range(len(env.transitions[i, j, a])):
                        s_prime = env.transitions[i, j, a][s][0]
                        r = env.transitions[i, j, a][s][1]
                        tr = env.transitions[i, j, a][s][3]
                        if s_prime is not None:
                            value_s_primes += tr * (r + gamma*value_est[s_prime[0], s_prime[1]])
                    Q[i, j, a] = value_s_primes
                value_est[i, j] = np.max(Q[i, j])
        policy = np.argmax(Q, axis=2)

    # Show the values and the policy
    env.clear_text
    env.draw_values(value_est)
    env.draw_actions(policy)
    env.render()
    sleep(1)

    # Save the state values and the policy
    fnames = "values.npy", "policy.npy"
    np.save(fnames[0], value_est)
    np.save(fnames[1], policy)
    print("Saved state values and policy to", *fnames)


"==================  TAsk 3  ================"

    # Reset the environment
    state = env.reset()

    # Compute state values and the policy
    # TODO: Compute the value function and policy (Tasks 1, 2 and 3)
    value_est, policy = np.zeros((env.w, env.h)), np.zeros((env.w, env.h))

    n_actions = env.transitions.shape[2]
    gamma = 0.9
    Q = np.zeros((env.w, env.h, n_actions))
    epsilon = 1e-4
    deltaV = [] # np.zeros((env.w, env.h))
    deltaQ = []
    v_last = np.zeros((env.w, env.h))
    p_last = np.zeros((env.w, env.h))
    for _ in range(100):
        for i in range(env.w):
            for j in range(env.h): #-1,-1,-1):
                for a in range(n_actions):
                    value_s_primes = 0
                    for s in range(len(env.transitions[i, j, a])):
                        s_prime = env.transitions[i, j, a][s][0]
                        r = env.transitions[i, j, a][s][1]
                        tr = env.transitions[i, j, a][s][3]
                        if s_prime is not None:
                            value_s_primes += tr * (r + gamma*value_est[s_prime[0], s_prime[1]])
                    Q[i, j, a] = value_s_primes
                value_est[i, j] = np.max(Q[i, j])
        policy = np.argmax(Q, axis=2)
        deltaV.append(np.sum(np.abs(value_est - v_last)))
        deltaQ.append(np.sum(np.abs(policy - p_last)))
        if(deltaV[-1] < epsilon):
            break
        # if(deltaQ[-1] < epsilon):
        #     break
        for i in range(env.w):
            for j in range(env.h):
                v_last[i, j] = value_est[i, j]
                p_last[i, j] = policy[i, j]
# if(not env.is_rocks(i, j) and (i is not env.harbour_x and j is not env.harbour_y)):
#     deltaV[i, j] = np.abs(value_est[i, j] - v_last[i, j])
#     if( deltaV[i, j] > max_deltaV ):
#         max_deltaV = deltaV[i, j]
# v_last[i, j] = value_est[i, j]

"==================  TAsk 4  ================"
    # Run a single episode
    # TODO: Run multiple episodes and compute the discounted returns (Task 4)

    return_history, timestep_history = [], []
    gamma = 0.9
    train_episodes = 1000
    for episode_number in range(train_episodes):
        return_ini_state, timesteps = 0, 0
        done = False
        # Reset the environment and observe the initial state
        state = env.reset()
        while not done:
            # Select a random action
            # TODO: Use the policy to take the optimal action (Task 2)
            # action = int(np.random.random()*4)
            # select action based on the policy obtained in previous tasks
            action = policy[state[0], state[1]]

            # Step the environment
            state, reward, done, _ = env.step(action)

            # Render and sleep
            # env.render()
            # sleep(0.5)
            return_ini_state += (gamma**timesteps) * reward
            timesteps += 1

        print("Episode {} finished. Discounted return: {:.3g} ({} timesteps)"
              .format(episode_number, return_ini_state, timesteps))

        # Bookkeeping (mainly for generating plots)
        return_history.append(return_ini_state)
        timestep_history.append(timesteps)

    avg = np.mean(return_history)
    sd = np.std(return_history)
    print(avg)
    print(sd)

    # Save the state values and the policy
    fnames = "return_history.npy"
    np.save(fnames, return_history)
