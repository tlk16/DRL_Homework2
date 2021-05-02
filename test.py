#!/usr/bin/env python
"""Run Atari Environment with DQN."""

import argparse
import os
import random
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.wrappers import wrap_deepmind
from deeprl_hw2.model import Model
from deeprl_hw2.memory import ReplayMemory
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('--seed', default=23333, type=int, help='Random seed')
    parser.add_argument('--memory_size', default=1000000, type=int, help='memory_size')

    args = parser.parse_args()

    seed_all(args.seed)

    env = gym.make(args.env)
    n_actions = env.action_space.n  # n = 6 for SpaceInvaders-v0
    env = wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=True, scale=False)  # todo
    env.reset()
    n_frame , _, _, _ = env.step(0)
    n_frame1,_,_,_ = env.step(0)
    print(n_frame[:,:,0])
    print(np.sum(n_frame[:,:,0]-n_frame[:,:,0]))
    plt.imshow(n_frame[:,:,1])
    plt.show()

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()