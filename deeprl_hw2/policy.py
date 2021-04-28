"""
RL Policy classes.
"""
import numpy as np
import random


class Policy:
    """
    Base class representing an MDP policy.
    Policies are used by the agent to choose actions.
    """

    def select_action(self, **kwargs):
        """Used by agents to select actions.

        Returns
        -------
        Any:
          An object representing the chosen action. Type depends on
          the hierarchy of policy instances.
        """
        raise NotImplementedError('This method should be overriden.')


class LinearDecayGreedyEpsilonPolicy(Policy):
    """Policy with a parameter that decays linearly.

    Like GreedyEpsilonPolicy but the epsilon decays from a start value
    to an end value over k steps.

    Parameters
    ----------
    start_value: int, float
      The initial value of the parameter
    end_value: int, float
      The value of the policy at the end of the decay.
    num_steps: int
      The number of steps over which to decay the value.

    """

    def __init__(self, n_actions, start_value, end_value, num_steps):
        self.n_actions = n_actions
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps

        self.epsilon = start_value
        self.delta = (start_value - end_value) / num_steps

    def select_action(self, q_values, is_training=True):
        """Decay parameter and select action.

        Parameters
        ----------
        q_values: np.array [1, num_actions] or [num_actions,]
          The Q-values for each action.
        is_training: bool, optional
          If true then parameter will be decayed. Defaults to true.

        Returns
        -------
        int:
          The action index chosen.
        """
        if is_training and self.epsilon > self.end_value:
            self.epsilon = self.epsilon - self.delta

        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return q_values.squeeze().argmax(axis=0).item()


    def reset(self):
        """Start the decay over at the start value."""
        self.epsilon = self.start_value
