"""
Random Copper Control Policy
Author: Hayden Hedman
Revised: 2025-12-10

A negative-control policy that selects actions randomly.
This represents a system applying copper exposure with
no adaptive feedback from bacterial resistance trends.

Biological purpose:
A random policy provides a baseline to compare against
structured policies (rule-based, Q-learning). It illustrates
how resistance evolves when copper exposure is not controlled.

Actions:
0 = decrease copper
1 = maintain copper
2 = increase copper
"""

import numpy as np


class RandomPolicy:
    """Stateless policy that selects copper actions at random."""

    def __init__(self, action_space):
        self.action_space = action_space

    def reset(self):
        """Random policy has no internal state; nothing to reset."""
        pass

    def act(self, state):
        """
        Select a random action each cycle.
        No information from the environment is used.
        """
        return self.action_space.sample()
