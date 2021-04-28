"""Main DQN agent."""

import copy
import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import gym


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

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit the model to the provided environment.

        Parameters
        ----------
        env: gym.Env
          wrapped Atari environment.
        num_iterations: int
          How many episodes to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets.
        """

        # Initialize --------------------------------------------------------------------------
        rewards = []  # record rewards for plot

        self.memory.clear()
        q1 = copy.deepcopy(self.q_network).cuda()
        q2 = copy.deepcopy(self.q_network).cuda()
        optimizer = torch.optim.RMSprop(q2.parameters(), lr=1e-4)  # todo: double DQN, lr TUNING
        # --------------------------------------------------------------------------------------

        for iteration in tqdm.tqdm(range(num_iterations)):
            state = env.reset()
            reward_per_episode = []
            for episode in range(max_episode_length):
                q_values = q1(torch.from_numpy(state.astype(np.float64)).cuda().unsqueeze(0).permute(0, 3, 1, 2))
                action = self.policy.select_action(q_values.cpu().detach().numpy())
                new_state, reward, done, info = env.step(action)
                self.memory.append(state, action, reward)
                reward_per_episode.append(reward)
                if done or (episode == max_episode_length - 1):
                    self.memory.end_episode(new_state, done)
                    rewards.append(np.mean(reward_per_episode))
                    break

            if (len(self.memory) > self.num_burn_in) and (iteration % self.train_freq == 0):

                train_samples = self.memory.sample(batch_size=self.batch_size)
                q1, q2 = self.train(q1, q2, train_samples, optimizer)

            if iteration % 50 == 0:
                plot_and_print(rewards, smooth_length=10)

    def train(self, q1, q2, train_samples, optimizer):
        """
        one gradient descent step for DQN
        :param q1: q_network with gradient descent;
        :param q2: target network
        :param train_samples: list of [st, at, rt, st+1(s')]
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
        q_target = rt + self.gamma * torch.max(q2(s_prime), dim=-1)[0]  # [batch_size]
        q_value = torch.gather(input=q1(st), dim=-1, index=at.unsqueeze(-1))
        loss = torch.mean((q_target.detach() - q_value) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # DQN: update the target network(q2) per target_update_freq steps
        if self.train_num % self.target_update_freq == 0:
            q2 = q1
        self.train_num += 1

        q1.eval()
        q2.eval()

        return q1, q2


def plot_and_print(rewards, smooth_length):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    l = smooth_length # len(train_loader)
    ax2.plot(np.convolve(np.array(rewards), np.ones(l) / l)[l:-l], color='r')
    print('rewards', '%.5g' % (np.mean(rewards[-l:])))
    ax2.legend(['rewards'], loc='upper right')
    plt.savefig('trainning_curve.png')
    plt.close()


def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        pass
