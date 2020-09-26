import numpy as np
import mab


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class IncentivizedMAB:
    def __init__(self, trials, mab_alg_name, mean_arm_reward, drift_func):
        self.trials = trials
        self.mab_alg_name = mab_alg_name
        self.cumulative_reward = dict()
        self.cumulative_compensation = dict()
        self.cumulative_regret = dict()
        self.mean_arm_reward = mean_arm_reward
        self.avg_reward = dict()
        self.avg_compensation = dict()
        self.avg_regret = dict()
        self.frequency = dict()
        self.drift_func = drift_func
        self.optimal_arm = None

        max_reward = -1
        for arm, reward in self.mean_arm_reward.items():
            if reward > max_reward:
                max_reward = reward
                self.optimal_arm = arm

        for arm in self.mean_arm_reward.keys():
            self.avg_reward[arm] = 0
            self.frequency[arm] = 0

        self.cumulative_compensation[0] = 0
        self.cumulative_regret[0] = 0

    def run(self):
        for t in range(1, self.trials + 1):
            principal_arm = self.get_principal_arm(t)
            player_arm = self.get_player_arm()

            reward, compensation = self.get_reward(principal_arm), 0
            if principal_arm != player_arm:
                compensation = self.get_compensation(principal_arm, player_arm)
                drift = self.get_drift(compensation)
                reward += drift

            if self.mab_alg_name == 'e-greedy':
                reward = sigmoid(reward)

            self.update_frequency(principal_arm)
            self.update_reward(reward, principal_arm)
            self.update_regret(t, reward)
            self.update_compensation(compensation, t)

    def get_principal_arm(self, t):
        if self.mab_alg_name == 'UCB':
            return mab.ucb(avg_rewards=self.avg_reward,
                           trial=t,
                           frequency=self.frequency)
        elif self.mab_alg_name == 'e-greedy':
            return mab.epsilon_greedy(avg_rewards=self.avg_reward, trial=t)
        elif self.mab_alg_name == 'thompson-sampling':
            return mab.thompson_sampling(avg_rewards=self.avg_reward, frequency=self.frequency)

    def get_player_arm(self):
        max_reward, best_arm = -1, 0
        for arm, reward in self.avg_reward.items():
            if reward > max_reward:
                max_reward = reward
                best_arm = arm
        return best_arm

    def update_reward(self, reward, principal_arm):
        if principal_arm in self.cumulative_reward:
            self.cumulative_reward[principal_arm] += reward
        else:
            self.cumulative_reward[principal_arm] = reward
        self.avg_reward[principal_arm] = self.cumulative_reward[principal_arm] / self.frequency[principal_arm]

    def update_frequency(self, principal_arm):
        self.frequency[principal_arm] += 1

    def update_compensation(self, compensation, t):
        self.cumulative_compensation[t] = compensation
        if t > 1:
            self.cumulative_compensation[t] += self.cumulative_compensation[t - 1]
        self.avg_compensation[t] = self.cumulative_compensation[t] / t

    def update_regret(self, t, reward):
        regret = self.avg_reward[self.optimal_arm] - reward
        # print ("DEBUG regret: ", regret, "opti: ", self.optimal_arm)
        self.cumulative_regret[t] = regret
        if t > 1:
            self.cumulative_regret[t] += self.cumulative_regret[t - 1]
        self.avg_regret[t] = self.cumulative_regret[t] / t

    def get_compensation(self, principal_arm, player_arm):
        return self.avg_reward[player_arm] - self.avg_reward[principal_arm]

    def get_drift(self, compensation):
        return self.drift_func(compensation)

    def get_reward(self, principal_arm):
        return np.random.normal(self.mean_arm_reward[principal_arm], 1)





