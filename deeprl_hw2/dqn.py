"""Main DQN agent."""

import copy
import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import gym

from deeprl_hw2.wrappers import wrap_deepmind


class DQNAgent:
    """Class implementing DQN.

    Parameters
    ----------
    q_network: nn.Model
        Q-network model.
    memory: deeprl_hw2.core.Memory
        Replay memory.
    gamma: float
        Discount factor.
    target_update_freq: int
      Update the target network per (target_update_freq) train steps
    num_burn_in: int
      Before begin updating the Q-network the replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      train the Q-Network per (train_freq) steps.
    batch_size: int
      How many samples in each mini-batch.
    """
    def __init__(self,
                 q_network,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size):
        self.q_network = q_network
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size

        self.train_num = 0  # toe record num of self.train is called

    def fit(self, env_raw, num_steps):
        """Fit the model to the provided environment.

        Parameters
        ----------
        env: gym.Env
          wrapped Atari environment.
        num_iterations: int
          How many episodes to perform.
        """

        # Initialize --------------------------------------------------------------------------
        losses = []   # record loss for plot

        self.memory.clear()
        self.policy.reset()
        env = wrap_deepmind(env_raw, frame_stack=True, episode_life=True, clip_rewards=True, scale=False)
        state = env.reset()
        q1 = copy.deepcopy(self.q_network.cpu()).cuda()  # For DQN， q1 is the online network
        q2 = copy.deepcopy(self.q_network.cpu()).cuda()  # For DQN, q2 is the target network
        optimizer = torch.optim.Adam(q1.parameters(), lr=0.0000625, eps=1.5e-4)  # todo: double DQN, lr TUNING
        # --------------------------------------------------------------------------------------

        for n_step in tqdm.tqdm(range(num_steps)):
            q_values = q1(torch.from_numpy(state.astype(np.float64)).cuda().unsqueeze(0).permute(0, 3, 1, 2))
            action = self.policy.select_action(q_values.cpu().detach().numpy())
            new_state, reward, done, info = env.step(action)
            self.memory.append(state, action, reward)
            state = new_state

            if done:
                self.memory.end_episode(new_state, done)
                state = env.reset()

            if (len(self.memory) > self.num_burn_in) and (n_step % self.train_freq == 0):
                train_samples = self.memory.sample(batch_size=self.batch_size)
                q1, q2, loss = self.train(q1, q2, train_samples, optimizer)
                losses.append(loss)

            if (n_step % 100000 == 0) and (len(self.memory) > self.num_burn_in):
                plot_and_print(losses)

            if n_step % 100000 == 0:
                evaluate_rewards = self.evaluate(env_raw, q1, num_episodes=20)
                print('evaluate', evaluate_rewards, evaluate_rewards.mean())

    def train(self, q1, q2, train_samples, optimizer):
        """
        one gradient descent step for DQN, update target network periodically
        :param q1: q_network with gradient descent;
        :param q2: target network
        :param train_samples: list of [st, at, rt, st+1(s'), terminal]
        :param optimizer torch.optim
        :return: q1, q2
        """
        q1.train()
        q2.train()
        st = torch.from_numpy(np.array([sample[0] for sample in train_samples]).astype(np.float64).transpose((0, 3, 1, 2))).cuda()
        # st [batch_size, 4, 84, 84]
        s_prime = torch.from_numpy(np.array([sample[3] for sample in train_samples]).astype(np.float64).transpose((0, 3, 1, 2))).cuda()
        # s_prime [batch_size, 4, 84, 84]
        at = torch.from_numpy(np.array([sample[1] for sample in train_samples]).astype(np.int64)).cuda()
        # at [batch_size,]
        rt = torch.from_numpy(np.array([sample[2] for sample in train_samples]).astype(np.float64)).cuda()
        # rt [batch_size,]
        terminal = torch.from_numpy(np.array([sample[4] for sample in train_samples]).astype(np.int64)).cuda()
        # terminal [batch_size,]

        # q_target = torch.where(terminal == 0,
        #                        rt + self.gamma * torch.max(q2(s_prime), dim=-1)[0],
        #                        rt)  # [batch_size]

        q_target = self.gamma * torch.max(q2(s_prime), dim=-1)[0] * (1. - terminal) + rt  # [batch_size]
        q_value = torch.gather(input=q1(st), dim=-1, index=at.unsqueeze(-1))  # [batch_size, 1]
        loss = torch.nn.SmoothL1Loss()(q_value.float(), q_target.detach().float().unsqueeze(1))

        # torch.mean((q_value - q_target.detach()) ** 2)

        optimizer.zero_grad()
        loss.backward()
        for param in q1.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()

        # DQN: update the target network(q2) per target_update_freq steps
        if self.train_num % self.target_update_freq == 0:
            q2.load_state_dict(q1.state_dict())
        self.train_num += 1

        q1.eval()
        q2.eval()

        return q1, q2, loss.cpu().item()

    def evaluate(self, env, q1, num_episodes):
        """Test your agent with a provided environment.
        """
        env = wrap_deepmind(env, frame_stack=True, episode_life=False, clip_rewards=False, scale=False)
        state = env.reset()

        rewards = np.zeros((num_episodes,))
        n_episode = 0
        while True:
            q_values = q1(torch.from_numpy(state.astype(np.float64)).cuda().unsqueeze(0).permute(0, 3, 1, 2))
            action = self.policy.select_action(q_values.cpu().detach().numpy(), is_training=False)
            new_state, reward, done, info = env.step(action)
            state = new_state
            rewards[n_episode] += reward
            # env.render()

            if done:
                state = env.reset()
                n_episode += 1
                if n_episode >= num_episodes:
                    break

        return rewards


def plot_and_print(losses):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    l = 50000  # len(train_loader)
    ax2.plot(np.convolve(np.array(losses), np.ones(l) / l)[l:-l], color='b')
    print('loss', '%.5g' % (np.mean(losses[-l:])))
    ax2.legend(['loss'], loc='upper right')
    plt.savefig('loss.png')
    plt.close()

