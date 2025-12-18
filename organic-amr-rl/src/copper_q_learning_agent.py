"""
Q-Learning Agent for Copper-Driven Resistance Dynamics
Author: Hayden Hedman
Revised: 2025-12-10

This agent learns a simple policy to adjust copper exposure to influence
the evolution of antibiotic resistance (modeled via MIC_chloro + MIC_polyB).

The design favors biological interpretability rather than ML complexity.
"""

import numpy as np


# ======================================================================
# TABULAR Q-LEARNING AGENT
# ======================================================================
class QLearner:
    """
    A simple coarse-discretized tabular Q-learning agent.

    The environment produces continuous values (copper, two MICs, cycle),
    which are mapped into bins for a lower-dimensional, interpretable
    Q-table. This is intentionally simple for microbiology audiences.
    """

    def __init__(
        self,
        env,
        learning_rate=0.1,
        discount=0.95,
        epsilon=0.10,
        copper_bins=10,
        mic_bins=10,
        cycle_bins=10
    ):

        self.env = env
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon

        # --------------------------------------------------------------
        # Discretization scheme (biologically interpretable bins)
        # --------------------------------------------------------------
        self.copper_bins = copper_bins
        self.mic_bins = mic_bins
        self.cycle_bins = cycle_bins

        # Precompute bin edges
        self.copper_edges = np.linspace(0, 100, copper_bins + 1)
        self.mic_edges = np.linspace(0, 512, mic_bins + 1)
        self.cycle_edges = np.linspace(0, env.max_cycles, cycle_bins + 1)

        # --------------------------------------------------------------
        # Q-table dimensions:
        # (copper_bin × chloroMIC_bin × polyMIC_bin × cycle_bin × action)
        # --------------------------------------------------------------
        self.q_table = np.zeros(
            (
                copper_bins,
                mic_bins,
                mic_bins,
                cycle_bins,
                env.action_space.n
            )
        )

    # ==================================================================
    # RESET (per-episode, Q-table persists)
    # ==================================================================
    def reset(self):
        """No internal state to reset; Q-table persists across episodes."""
        pass

    # ==================================================================
    # DISCRETIZATION UTILITIES
    # ==================================================================
    def _bin(self, value, edges):
        """Return the bin index for any continuous value."""
        return int(np.digitize(value, edges) - 1)

    def discretize(self, state):
        """
        Convert environment state → Q-table index.

        State format:
        [ copper, MIC_chloro, MIC_polyB, cycle, growth_inhibition ]
        """
        copper, mic_c, mic_p, cycle, _ = state

        return (
            self._bin(copper, self.copper_edges),
            self._bin(mic_c, self.mic_edges),
            self._bin(mic_p, self.mic_edges),
            self._bin(cycle, self.cycle_edges)
        )

    # ==================================================================
    # ACTION SELECTION (ε-greedy)
    # ==================================================================
    def act(self, state):
        """Choose an action using ε-greedy exploration."""
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        ds = self.discretize(state)
        return int(np.argmax(self.q_table[ds]))

    # ==================================================================
    # CORE Q-LEARNING UPDATE
    # ==================================================================
    def learn(self, state, action, reward, next_state, done):
        ds = self.discretize(state)
        ds_next = self.discretize(next_state)

        current_q = self.q_table[ds][action]
        max_future_q = 0 if done else np.max(self.q_table[ds_next])

        updated_q = current_q + self.lr * (reward + self.gamma * max_future_q - current_q)
        self.q_table[ds][action] = updated_q

    # ==================================================================
    # OPTIONAL STANDALONE TRAIN LOOP (not used in run_experiments.py)
    # ==================================================================
    def train(self, episodes=200):
        """
        Optional standalone training loop.
        run_experiments.py handles training automatically during stepping.
        """
        reward_history = []

        for ep in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                self.learn(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state

            reward_history.append(total_reward)
            print(f"Episode {ep+1}/{episodes} | Total Reward: {total_reward:.2f}")

        return reward_history
