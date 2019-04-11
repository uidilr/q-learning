import numpy as np
import gym
import argparse


def get_args():
    """
    get argument
    """
    parser = argparse.ArgumentParser('Q-learning')
    parser.add_argument('--alpha', '-a', default=0.1, type=float, help='alpha for Q-learning')
    parser.add_argument('--gamma', '-g', default=0.95, type=float, help='gamma for Q-learning')
    return parser.parse_args()


def epssilon_greeedy(Q, obs, eps=0.1):
    if np.random.random() < eps:
        n_actions = Q.shape[-1]
        return np.random.randint(n_actions)
    return np.argmax(Q[obs])


if __name__ == '__main__':
    args = get_args()

    # prepare environment
    from gym.envs.toy_text import FrozenLakeEnv
    env = FrozenLakeEnv(is_slippery=False)
    n_states, n_actions = env.observation_space.n, env.action_space.n
    max_episodes = 10000
    max_steps = 1000
    alpha = args.alpha
    gamma = args.gamma

    Q = np.zeros(shape=(n_states, n_actions))

    for n_episode in range(max_episodes):
        obs = env.reset()
        epi_rewards = []
        for step in range(max_steps):
            a = epssilon_greeedy(Q, obs=obs, eps=0.3)
            next_obs, reward, done, info = env.step(a)
            epi_rewards.append(reward * (gamma ** step))
            Q[obs, a] = Q[obs, a] + alpha * (reward + gamma * np.max(Q[next_obs]) - Q[obs, a])

            if done:
                print("step: {:3d}".format(step), "  rewards: ", sum(epi_rewards))
                break
            obs = next_obs

    # 0: LEFT, 1: DOWN, 2: RIGHT, 3: UP
    int_arrow_dict = {0: '\u2190', 1: '\u2193', 2: '\u2192', 3: '\u2191'}
    # print learned policy
    print(np.vectorize(int_arrow_dict.get)(np.argmax(Q, axis=-1)).reshape(int(n_states ** (1/2)),
                                                                          int(n_states ** (1/2))))
