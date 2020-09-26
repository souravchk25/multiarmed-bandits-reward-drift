import incentivized_mab
import matplotlib.pyplot as plt
import numpy as np

arm_rewards = {
    1: 0.9,
    2: 0.8,
    3: 0.7,
    4: 0.6,
    5: 0.5,
    6: 0.4,
    7: 0.3,
    8: 0.2,
    9: 0.1,
}

TRIALS = 20000

ucb = incentivized_mab.IncentivizedMAB(trials=TRIALS, mab_alg_name='UCB', mean_arm_reward=arm_rewards, coeff=1)
e_greedy = incentivized_mab.IncentivizedMAB(trials=TRIALS, mab_alg_name='e-greedy', mean_arm_reward=arm_rewards, coeff=1)
thompson = incentivized_mab.IncentivizedMAB(trials=TRIALS, mab_alg_name='thompson-sampling', mean_arm_reward=arm_rewards, coeff=1)

ucb.run()
e_greedy.run()
thompson.run()

y1 = [ucb.cumulative_regret[t] for t in range(1, TRIALS + 1)]
plt.plot([i for i in range(1, TRIALS + 1)], y1, 'r', linestyle='-', label='UCB Regret')

y2 = [ucb.cumulative_compensation[t] for t in range(1, TRIALS + 1)]
plt.plot([i for i in range(1, TRIALS + 1)], y2, 'b', linestyle='-', label='UCB Compensation')

y3 = [e_greedy.cumulative_regret[t] for t in range(1, TRIALS + 1)]
plt.plot([i for i in range(1, TRIALS + 1)], y3, 'r', linestyle='--', label='e-Greedy Regret')

y4 = [e_greedy.cumulative_compensation[t] for t in range(1, TRIALS + 1)]
plt.plot([i for i in range(1, TRIALS + 1)], y4, 'b', linestyle='--', label='e-Greedy Compensation')

y5 = [thompson.cumulative_regret[t] for t in range(1, TRIALS + 1)]
plt.plot([i for i in range(1, TRIALS + 1)], y5, 'r', linestyle='-.', label='Thompson Regret')

y6 = [thompson.cumulative_compensation[t] for t in range(1, TRIALS + 1)]
plt.plot([i for i in range(1, TRIALS + 1)], y6, 'b', linestyle='-.', label='Thompson Compensation')

plt.title('Trend of Regret and Compensation')
plt.xlabel('Turns')
plt.ylabel('Regret and Compensation')
plt.legend(loc='upper left')
plt.savefig('plot.png')





