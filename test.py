import incentivized_mab
import matplotlib.pyplot as plt

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

TRIALS = 200000

ucb = incentivized_mab.IncentivizedMAB(trials=TRIALS, mab_alg_name='UCB', mean_arm_reward=arm_rewards, coeff=1)
e_greedy = incentivized_mab.IncentivizedMAB(trials=TRIALS, mab_alg_name='e-greedy', mean_arm_reward=arm_rewards, coeff=1)

ucb.run()
e_greedy.run()

y1 = [ucb.cumulative_regret[t] for t in range(1, TRIALS + 1)]
plt.plot([i for i in range(1, TRIALS + 1)], y1, 'r', linestyle='-')

y2 = [ucb.cumulative_compensation[t] for t in range(1, TRIALS + 1)]
plt.plot([i for i in range(1, TRIALS + 1)], y2, 'b', linestyle='-')

y3 = [e_greedy.cumulative_regret[t] for t in range(1, TRIALS + 1)]
plt.plot([i for i in range(1, TRIALS + 1)], y3, 'r', linestyle='--')

y4 = [e_greedy.cumulative_compensation[t] for t in range(1, TRIALS + 1)]
plt.plot([i for i in range(1, TRIALS + 1)], y4, 'b', linestyle='--')

plt.show()




