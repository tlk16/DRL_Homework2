#!/usr/bin/env python
"""Run Atari Environment with DQN."""

import argparse
import os
import random
import numpy as np
import gym
import torch

from deeprl_hw2.dqn import DQNAgent
from deeprl_hw2.wrappers import wrap_deepmind, make_atari
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

    env = make_atari('SpaceInvadersNoFrameskip-v4')

    n_actions = env.action_space.n  # n = 6 for SpaceInvaders-v0

    model = Model(in_channels=4, n_actions=n_actions)
    memory = ReplayMemory(max_size=args.memory_size)
    policy = LinearDecayGreedyEpsilonPolicy(n_actions=n_actions, start_value=1, end_value=0.1, num_steps=1000000)
    agent = DQNAgent(q_network=model, memory=memory, gamma=0.99, target_update_freq=1500,
                     num_burn_in=100000, batch_size=128, policy=policy, train_freq=8)
    agent.fit(env, num_steps=50000000)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
