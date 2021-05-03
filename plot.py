import numpy as np
import matplotlib.pyplot as plt
import gym
import os
import torch
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.model import DQN, LinearQN, DuelDQN
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy
from deeprl_hw2.wrappers import wrap_deepmind, make_atari


for n_question in [2, 3, 4, 5, 6, 7]:
    final_path = 'Question' + str(n_question) + '/rewards_final.npy'
    rewards_final = np.load(final_path)
    print(n_question, rewards_final.shape, rewards_final.std(), rewards_final.mean())

    curve_path = 'Question' + str(n_question) + '/rewards.npy'
    rewards_curve = np.load(curve_path)
    plt.figure()
    plt.plot([i * 100000 for i in range(len(rewards_curve))], rewards_curve, color='r')
    plt.xlabel('Train steps')
    plt.ylabel('Rewards per episode')
    plt.legend(['Rewards'])
    plt.savefig(str(n_question) + '.png', dpi=300)


def evaluate_video(env, q1, num_episodes, save_name, policy):
    """Test your agent with a provided environment.
    """
    env = wrap_deepmind(env, frame_stack=True, episode_life=False, clip_rewards=False, scale=False)
    os.mkdir(save_name)  # n_step is just used to specify the save dir
    env = gym.wrappers.Monitor(env, save_name)
    state = env.reset()

    rewards = np.zeros((num_episodes,))
    n_episode = 0
    while True:
        q_values = q1(torch.from_numpy(state.astype(np.float64)).cuda().unsqueeze(0).permute(0, 3, 1, 2))
        action = policy.select_action(q_values.cpu().detach().numpy(), is_training=False)
        new_state, reward, done, info = env.step(action)
        state = new_state
        rewards[n_episode] += reward
        env.render()

        if done:
            n_episode += 1
            if n_episode >= num_episodes:
                break
            state = env.reset()

    return rewards


env = make_atari('SpaceInvadersNoFrameskip-v4')
n_actions = env.action_space.n
policy = LinearDecayGreedyEpsilonPolicy(n_actions=n_actions, start_value=1, end_value=0.1, num_steps=1000000)
q1 = DuelDQN(in_channels=4, n_actions=n_actions).cuda()
# q1.load_state_dict(torch.load('Question4/model.pth'))

evaluate_video(env, q1, num_episodes=1, save_name='q7_0', policy=policy)




