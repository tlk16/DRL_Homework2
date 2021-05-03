import numpy as np
import matplotlib.pyplot as plt

for n_question in [2, 3, 4, 5, 6]:
    final_path = 'Question' + str(n_question) + '/rewards_final.npy'
    rewards_final = np.load(final_path)
    print(n_question, rewards_final.shape, rewards_final.std(), rewards_final.mean())

    curve_path = 'Question' + str(n_question) + '/rewards.npy'
    rewards_curve = np.load(curve_path)
    plt.figure()
    plt.plot(rewards_curve)
    plt.savefig(str(n_question) + '.png')
