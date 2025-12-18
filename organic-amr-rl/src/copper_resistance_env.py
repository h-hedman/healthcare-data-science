"""
Copper Exposure & Resistance Simulation Environment
Author: Hayden Hedman
Date: 2025-11-01

Self-contained RL environment modeling copper-driven evolution
of antibiotic resistance (Li et al., 2021; inspired by Weaver et al., 2024).

This script is intentionally standalone so that other modules such as:
 - run_experiments.py
 - random_policy.py
 - rule_based_agent.py
 - copper_q_learning_agent.py

can import:
     from scripts.copper_resistance_env import CopperResistanceEnv
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CopperResistanceEnv(gym.Env):
    """Environment for copper-driven antibiotic resistance dynamics."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_cycles=40):
        super().__init__()

        # ---------------------------------------------------------------
        # ACTION SPACE (3 discrete control moves)
        # ---------------------------------------------------------------
        # 0: decrease copper
        # 1: maintain copper
        # 2: increase copper
        self.action_space = spaces.Discrete(3)

        # ---------------------------------------------------------------
        # OBSERVATION SPACE
        # [copper mg/L, MIC_chloro, MIC_polyB, cycle, growth_inhibition]
        # ---------------------------------------------------------------
        low = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([100, 512, 512, max_cycles, 1], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Biological parameters
        self.max_cycles = max_cycles
        self.copper_step = 5
        self.reversal_scale = 0.02
        self.mutation_scale = 0.10
        self.max_MIC_sum = 512 + 512

        # Initialize system state
        self.reset()

    # ===================================================================
    # RESET
    # ===================================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initial conditions approximate Li et al. (2021) sublethal copper setup
        self.cycle = 0
        self.copper = 10.0
        self.MIC_chloro = 2.0
        self.MIC_polyB = 2.0
        self.growth_inhibition = 0.0

        return self._get_state(), {}

    # ===================================================================
    # STEP FUNCTION
    # ===================================================================
    def step(self, action):

        # Advance cycle
        self.cycle += 1

        # ---------------------------------------------------------------
        # ACTION LOGIC: copper up/down/maintain
        # ---------------------------------------------------------------
        if action == 0:  # decrease
            self.copper = max(0, self.copper - self.copper_step)
        elif action == 2:  # increase
            self.copper = min(100, self.copper + self.copper_step)
        # action == 1 → maintain

        # ---------------------------------------------------------------
        # Resistance Dynamics (conceptual model)
        # ---------------------------------------------------------------
        pressure = self.copper / 100.0  # normalized copper selection pressure

        # Mutation-driven MIC increases under high copper
        if np.random.rand() < pressure:
            self.MIC_chloro += np.random.uniform(0.5, 3.0)
            self.MIC_polyB += np.random.uniform(0.2, 2.0)

        # Small phenotypic reversals if copper sufficiently low
        if self.copper < 10:
            self.MIC_chloro -= np.random.uniform(0, 0.5)
            self.MIC_polyB -= np.random.uniform(0, 0.2)

        # Clamp MIC values
        self.MIC_chloro = max(0.0, self.MIC_chloro)
        self.MIC_polyB = max(0.0, self.MIC_polyB)

        # ---------------------------------------------------------------
        # Growth Phenotype (simple inhibition proxy)
        # ---------------------------------------------------------------
        antibiotic_pressure = (self.MIC_chloro + self.MIC_polyB) / self.max_MIC_sum
        self.growth_inhibition = max(0.0, 1.0 - antibiotic_pressure)

        # ---------------------------------------------------------------
        # Reward Function
        # Encourages:
        #  - high growth (low resistance)
        # Penalizes:
        #  - copper usage
        #  - resistance
        # ---------------------------------------------------------------
        reward = (
            2.0 * self.growth_inhibition
            - 0.01 * self.MIC_chloro
            - 0.01 * self.MIC_polyB
            - 0.05 * self.copper
        )

        done = self.cycle >= self.max_cycles

        return self._get_state(), reward, done, False, {}

    # ===================================================================
    # STATE PACKING
    # ===================================================================
    def _get_state(self):
        return np.array([
            self.copper,
            self.MIC_chloro,
            self.MIC_polyB,
            self.cycle,
            self.growth_inhibition
        ], dtype=np.float32)

    # ===================================================================
    # RENDER (optional)
    # ===================================================================
    def render(self):
        print(
            f"Cycle {self.cycle} | Cu={self.copper:.1f} mg/L | "
            f"MIC_chl={self.MIC_chloro:.2f} | MIC_polyB={self.MIC_polyB:.2f} | "
            f"Growth={self.growth_inhibition:.2f}"
        )
# ------------------------------------------------------------------------------------------------