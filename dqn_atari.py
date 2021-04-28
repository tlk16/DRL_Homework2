#!/usr/bin/env python
"""Run Atari Environment with DQN."""

import argparse
import os
import random
import numpy as np
import gym
import torch

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
    env = wrap_deepmind(env, episode_life=False, clip_rewards=True, frame_stack=True, scale=False)
    model = Model(in_channels=4, n_actions=n_actions)
    memory = ReplayMemory(max_size=args.memory_size)
    policy = LinearDecayGreedyEpsilonPolicy(n_actions=n_actions, start_value=1, end_value=0.1, num_steps=500000)
    agent = DQNAgent(q_network=model, memory=memory, gamma=0.99, target_update_freq=100,
                     num_burn_in=50000, batch_size=64, policy=policy, train_freq=1)
    agent.fit(env, num_iterations=5000000, max_episode_length=500)


if __name__ == '__main__':
    main()
