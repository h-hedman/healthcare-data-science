"""
Reward Sensitivity Analysis for Q-Learning Agent
Author: Hayden Hedman
Revised: 2026-03

Purpose:
Evaluate robustness of Q-learning performance under
alternative reward weight configurations.

Outputs:
- reward_sensitivity_results.csv
"""

import os
import time
import numpy as np
import pandas as pd

from copper_resistance_env import CopperResistanceEnv
from copper_q_learning_agent import QLearner


# ==============================================================
# CONFIGURATION
# ==============================================================

RANDOM_SEED = 88
EPISODES = 150
MAX_CYCLES = 40

# Baseline weights correspond exactly to original environment defaults
REWARD_CONFIGS = [
    {
        "label": "Baseline",
        "growth_weight": 2.0,
        "mic_weight": 0.01,
        "copper_weight": 0.05
    },
    {
        "label": "High Copper Penalty",
        "growth_weight": 2.0,
        "mic_weight": 0.01,
        "copper_weight": 0.10
    },
    {
        "label": "Low MIC Penalty",
        "growth_weight": 2.0,
        "mic_weight": 0.005,
        "copper_weight": 0.05
    }
]


# ==============================================================
# METRIC FUNCTIONS
# ==============================================================

def compute_auc(values):
    return np.trapz(values)


def run_reward_config(config):

    np.random.seed(RANDOM_SEED)

    env = CopperResistanceEnv(
        max_cycles=MAX_CYCLES,
        growth_weight=config["growth_weight"],
        mic_weight=config["mic_weight"],
        copper_weight=config["copper_weight"]
    )

    agent = QLearner(
        env,
        learning_rate=0.1,
        discount=0.95,
        epsilon=0.1
    )

    auc_chloro = []
    auc_poly = []
    copper_totals = []

    for ep in range(EPISODES):

        state, _ = env.reset()
        agent.reset()

        done = False
        chloro_vals = []
        poly_vals = []
        copper_vals = []

        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)

            copper, mic_c, mic_p, cycle, growth = next_state

            chloro_vals.append(mic_c)
            poly_vals.append(mic_p)
            copper_vals.append(copper)

            state = next_state

        auc_chloro.append(compute_auc(chloro_vals))
        auc_poly.append(compute_auc(poly_vals))
        copper_totals.append(np.sum(copper_vals))

    return {
        "Reward Configuration": config["label"],
        "Growth Weight": config["growth_weight"],
        "MIC Penalty Weight": config["mic_weight"],
        "Copper Penalty Weight": config["copper_weight"],
        "AUC MIC (Chloramphenicol), Mean": np.mean(auc_chloro),
        "AUC MIC (Chloramphenicol), SD": np.std(auc_chloro),
        "AUC MIC (Polymyxin B), Mean": np.mean(auc_poly),
        "AUC MIC (Polymyxin B), SD": np.std(auc_poly),
        "Copper Burden, Mean": np.mean(copper_totals),
        "Copper Burden, SD": np.std(copper_totals),
    }


# ==============================================================
# MAIN EXECUTION
# ==============================================================

def run_reward_sensitivity():

    start_time = time.time()
    print("\nStarting Reward Sensitivity Analysis...\n")

    results = []

    for config in REWARD_CONFIGS:
        print(f"Running configuration: {config['label']}")
        metrics = run_reward_config(config)
        results.append(metrics)

    summary_df = pd.DataFrame(results)
    # Round numeric columns for publication-ready output
    summary_df = summary_df.round(2)

    # ----------------------------------------------------------
    # Dynamic Output Directory Handling
    # ----------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(src_dir, "outputs", "summary_tables")

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "reward_sensitivity_results.csv"
    )

    summary_df.to_csv(output_path, index=False)

    runtime = time.time() - start_time

    print("\nReward sensitivity analysis complete.")
    print(f"Results saved to: {output_path}")
    print(f"Total runtime: {runtime:.2f} seconds\n")


# ==============================================================
if __name__ == "__main__":
    run_reward_sensitivity()