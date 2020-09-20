import numpy as np


def ucb(avg_rewards, trial, frequency):
    max_reward, best_arm = -1, 1
    if trial < len(avg_rewards):
        return trial
    for arm, reward in avg_rewards.items():
        if reward > max_reward:
            if frequency[arm] > 0:
                bound = 2 * np.log(trial) / frequency[arm]
            else:
                bound = 0
            max_reward = reward + np.sqrt(bound)
            best_arm = arm
    return best_arm


def epsilon_greedy(avg_rewards, trial):
    c, num_arms = 400, len(avg_rewards)
    epsilon = min(1, (c * num_arms) / trial)
    toss = np.random.random()
    if toss < epsilon:
        return np.random.choice([arm for arm in avg_rewards.keys()])
    else:
        best_arm, max_reward = -1, -1
        for arm, reward in avg_rewards.items():
            if reward > max_reward:
                max_reward = reward
                best_arm = arm
        return best_arm
