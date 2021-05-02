"""Replay Memory"""

import random


class Sample:
    # todo: unused
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """
    def __init__(self, s, a, r, s_prime, terminal):
        self.s = s
        self.a = a
        self.r = r
        self.s_prime = s_prime
        self.terminal = terminal


class ReplayMemory:
    """Interface for replay memories.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        self.next_index = 0  # the next index to be inserted
        self.max_size = max_size  # max_size of replay memory
        self.memory = [[] for _ in range(max_size)]  # a list as a ring buffer

    def append(self, state, action, reward):
        """
        append st, at, rt to memory
        :param state: np.array [84, 84, 4]
        :param action:
        :param reward:
        :return:
        """
        self.memory[self.next_index] = [state, action, reward]
        self.next_index = (self.next_index + 1) % self.max_size

    def end_episode(self, final_state, is_terminal):
        self.memory[self.next_index] = [final_state, is_terminal]
        # todo: termial will be judged by len(self.memory[self.next_index])
        self.next_index = (self.next_index + 1) % self.max_size

    def sample(self, batch_size):
        if (batch_size == 1) and self.max_size == 2:
            # todo: no memory in fact
            index = self.next_index
            if not (len(self.memory[index]) == 3):
                index = (index + 1) % self.max_size  # todo
            terminal = len(self.memory[(index + 1) % self.max_size]) == 2
            return [self.memory[index] + [self.memory[(index + 1) % self.max_size][0], terminal]]

        if len(self.memory[-1]) > 0:
            # the memory is full, so each index is legal for sampling
            max_index = self.max_size - 1 - 1
        else:
            max_index = self.next_index - 1 - 1
        samples = []
        while len(samples) < batch_size:
            index = random.randint(0, max_index)
            if (len(self.memory[index]) == 3) and ((index + 1) % self.max_size != self.next_index):
                terminal = len(self.memory[(index + 1) % self.max_size]) == 2
                samples.append(self.memory[index] + [self.memory[(index + 1) % self.max_size][0], terminal])
                # [st, at, rt] + [st+1, terminal])
        return samples

    def clear(self):
        self.memory = [[] for _ in range(self.max_size)]
        self.next_index = 0

    def __len__(self):
        if len(self.memory[-1]) > 0:
            return self.max_size
        else:
            return self.next_index
