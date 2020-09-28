import numpy as np
import constants


class MultiWeightAlg:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.weights = np.array(self.num_arms * [1.0])
        self.prob_dist = np.array(self.num_arms * [(1.0 / self.num_arms)])
        self.rate = constants.MULTI_WGT_ALG_LEARNING_RATE

    def get_arm(self):
        return np.random.choice(np.arange(1, self.num_arms + 1), p=self.prob_dist)

    def update_prob_dist(self):
        total_wgt = np.sum(self.weights)
        self.prob_dist = self.weights / total_wgt

    def update_weights(self, arm, reward):
        self.weights[arm - 1] = self.weights[arm - 1] * (1.0 + (self.rate * reward))
        self.update_prob_dist()


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


def thompson_sampling(avg_rewards, frequency):
    theta = dict()
    for arm, reward in avg_rewards.items():
        theta[arm] = np.random.normal(loc=reward, scale=np.sqrt((1.0 / (frequency[arm] + 1))))
    best_arm, max_reward = -1, theta[1]
    for arm in theta.keys():
        if theta[arm] >= max_reward:
            max_reward = theta[arm]
            best_arm = arm
    return best_arm


def boltzmann_sampling(avg_rewards, turn):
    total, dist, learning_rate = 0.0, [], (constants.BOLTZMANN_SAMPLING_LEARNING_RATE * 1.0 / turn)
    for arm, reward in avg_rewards.items():
        total += np.exp(reward / learning_rate)
    for arm, reward in avg_rewards.items():
        dist.append(np.exp(reward / learning_rate) / total)
    return np.random.choice(np.arange(1, len(avg_rewards) + 1), p=np.array(dist))


class PursuitAlg:
    def __init__(self, num_arms):
        self.arms = np.arange(1, num_arms + 1)
        self.dist = np.array(num_arms * [1.0 / num_arms])

    def update_prob_dist(self, avg_rewards):
        best_arm, max_reward = 1, -1
        for arm, reward in avg_rewards.items():
            if reward >= max_reward:
                max_reward = reward
                best_arm = arm
        for arm in avg_rewards.keys():
            if arm == best_arm:
                self.dist[arm - 1] += constants.PURSUIT_BETA * (1 - self.dist[arm - 1])
            else:
                self.dist[arm - 1] += constants.PURSUIT_BETA * (0 - self.dist[arm - 1])

    def get_arm(self):
        return np.random.choice(self.arms, p=self.dist)


class ReinforcementComparison:
    def __init__(self, num_arms):
        self.arms = np.arange(1, num_arms + 1)
        self.pi = np.array(num_arms * [0.0])
        self.avg_reward = 0

    def get_arm(self):
        total = np.sum(np.exp(self.pi))
        prob_dist = np.exp(self.pi) / total
        # print ("total: {}, pi: {}, dist: {}".format(total, self.pi, prob_dist))
        return np.random.choice(self.arms, p=prob_dist)

    def update_pi(self, arm, reward):
        self.pi[arm - 1] += constants.REINFORCEMENT_COMPARISON_BETA * (reward - self.avg_reward)
        self.update_avg_reward(reward)

    def update_avg_reward(self, reward):
        self.avg_reward = (1 - constants.REINFORCEMENT_COMPARISON_ALPHA) * self.avg_reward
        self.avg_reward += constants.REINFORCEMENT_COMPARISON_ALPHA * reward

