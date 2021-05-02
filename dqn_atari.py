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
from deeprl_hw2.model import DQN, DuelDQN, LinearQN
from deeprl_hw2.memory import ReplayMemory
from deeprl_hw2.policy import LinearDecayGreedyEpsilonPolicy


def seed_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description='Run DQN on Atari SpaceInvaders')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument('--seed', default=23333, type=int, help='Random seed')
    parser.add_argument('--memory_size', default=1000000, type=int, help='memory_size')
    parser.add_argument('--target_type', default='fixing', help='no-fixing | fixing | double')
    parser.add_argument('--model', default='DQN', help='Linear | DQN | Dueling')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')

    args = parser.parse_args()

    seed_all(args.seed)

    env = make_atari('SpaceInvadersNoFrameskip-v4')

    n_actions = env.action_space.n  # n = 6 for SpaceInvaders-v0

    if args.model == 'DQN':
        model = DQN(in_channels=4, n_actions=n_actions)
    elif args.model == 'Linear':
        model = LinearQN(in_channels=4, n_actions=n_actions)
    else:
        assert args.model == 'Dueling'
        model = DuelDQN(in_channels=4, n_actions=n_actions)

    memory = ReplayMemory(max_size=args.memory_size)
    policy = LinearDecayGreedyEpsilonPolicy(n_actions=n_actions, start_value=1, end_value=0.1, num_steps=1000000)
    agent = DQNAgent(q_network=model, memory=memory, gamma=0.99, target_update_freq=2500,
                     num_burn_in=50000, batch_size=args.batch_size, policy=policy, train_freq=4, target_type=args.target_type)
    agent.fit(env, num_steps=6000000)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
