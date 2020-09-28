import matplotlib.pyplot as plt
import numpy as np
from src import constants, incentivized_mab


def log(x):
    return np.log(1 + x)


def sqrt(x):
    return np.sqrt(x)


def linear(x):
    return x


def abs(x):
    return np.abs(x)


def quadratic(x):
    return x ** 2


ucb = incentivized_mab.IncentivizedMAB(trials=constants.TRIALS,
                                       mab_alg_name=constants.UCB,
                                       mean_arm_reward=constants.arm_rewards,
                                       drift_func=linear)

e_greedy = incentivized_mab.IncentivizedMAB(trials=constants.TRIALS,
                                            mab_alg_name=constants.E_GREEDY,
                                            mean_arm_reward=constants.arm_rewards,
                                            drift_func=linear)

thompson = incentivized_mab.IncentivizedMAB(trials=constants.TRIALS,
                                            mab_alg_name=constants.THOMPSON_SAMPLING,
                                            mean_arm_reward=constants.arm_rewards,
                                            drift_func=linear)

mwa = incentivized_mab.IncentivizedMAB(trials=constants.TRIALS,
                                       mab_alg_name=constants.MULTI_WGT_ALG,
                                       mean_arm_reward=constants.arm_rewards,
                                       drift_func=sqrt)

boltzmann = incentivized_mab.IncentivizedMAB(trials=constants.TRIALS,
                                             mab_alg_name=constants.BOLTZMANN_SAMPLING,
                                             mean_arm_reward=constants.arm_rewards,
                                             drift_func=sqrt)


pursuit = incentivized_mab.IncentivizedMAB(trials=constants.TRIALS,
                                           mab_alg_name=constants.PURSUIT_ALG,
                                           mean_arm_reward=constants.arm_rewards,
                                           drift_func=linear)

reinforcement_comparison = incentivized_mab.IncentivizedMAB(trials=constants.TRIALS,
                                                            mab_alg_name=constants.REINFORCEMENT_COMPARISON,
                                                            mean_arm_reward=constants.arm_rewards,
                                                            drift_func=linear)

# ucb.run()
# e_greedy.run()
# thompson.run()
# mwa.run()
# boltzmann.run()
# pursuit.run()
reinforcement_comparison.run()

y1 = [reinforcement_comparison.cumulative_regret[t] for t in range(1, constants.TRIALS + 1)]
plt.plot([i for i in range(1, constants.TRIALS + 1)], y1, 'r', linestyle='-', label='reinforcement_comparison Regret')

y2 = [reinforcement_comparison.cumulative_compensation[t] for t in range(1, constants.TRIALS + 1)]
plt.plot([i for i in range(1, constants.TRIALS + 1)], y2, 'b', linestyle='-', label='reinforcement_comparison Compensation')


#
# y1 = [pursuit.cumulative_regret[t] for t in range(1, constants.TRIALS + 1)]
# plt.plot([i for i in range(1, constants.TRIALS + 1)], y1, 'r', linestyle='-', label='pursuit Regret')
#
# y2 = [pursuit.cumulative_compensation[t] for t in range(1, constants.TRIALS + 1)]
# plt.plot([i for i in range(1, constants.TRIALS + 1)], y2, 'b', linestyle='-', label='pursuit Compensation')


# y1 = [boltzmann.cumulative_regret[t] for t in range(1, constants.TRIALS + 1)]
# plt.plot([i for i in range(1, constants.TRIALS + 1)], y1, 'r', linestyle='-', label='boltzmann Regret')
#
# y2 = [boltzmann.cumulative_compensation[t] for t in range(1, constants.TRIALS + 1)]
# plt.plot([i for i in range(1, constants.TRIALS + 1)], y2, 'b', linestyle='-', label='boltzmann Compensation')

# y1 = [mwa.cumulative_regret[t] for t in range(1, constants.TRIALS + 1)]
# plt.plot([i for i in range(1, constants.TRIALS + 1)], y1, 'r', linestyle='-', label='MWA Regret')
#
# y2 = [mwa.cumulative_compensation[t] for t in range(1, constants.TRIALS + 1)]
# plt.plot([i for i in range(1, constants.TRIALS + 1)], y2, 'b', linestyle='-', label='MWA Compensation')


# y1 = [ucb.cumulative_regret[t] for t in range(1, constants.TRIALS + 1)]
# plt.plot([i for i in range(1, constants.TRIALS + 1)], y1, 'r', linestyle='-', label='UCB Regret')
#
# y2 = [ucb.cumulative_compensation[t] for t in range(1, constants.TRIALS + 1)]
# plt.plot([i for i in range(1, constants.TRIALS + 1)], y2, 'b', linestyle='-', label='UCB Compensation')
#
# y3 = [e_greedy.cumulative_regret[t] for t in range(1, constants.TRIALS + 1)]
# plt.plot([i for i in range(1, constants.TRIALS + 1)], y3, 'r', linestyle='--', label='e-Greedy Regret')
#
# y4 = [e_greedy.cumulative_compensation[t] for t in range(1, constants.TRIALS + 1)]
# plt.plot([i for i in range(1, constants.TRIALS + 1)], y4, 'b', linestyle='--', label='e-Greedy Compensation')
#
# y5 = [thompson.cumulative_regret[t] for t in range(1, constants.TRIALS + 1)]
# plt.plot([i for i in range(1, constants.TRIALS + 1)], y5, 'r', linestyle='-.', label='Thompson Regret')
#
# y6 = [thompson.cumulative_compensation[t] for t in range(1, constants.TRIALS + 1)]
# plt.plot([i for i in range(1, constants.TRIALS + 1)], y6, 'b', linestyle='-.', label='Thompson Compensation')


plt.title('Trend of Regret and Compensation (Drift Func: Linear)')
plt.xlabel('Turns')
plt.ylabel('Regret and Compensation')
plt.legend(loc='upper left')
plt.savefig('plots/plot-linear-reinforcement_comparison.png')
