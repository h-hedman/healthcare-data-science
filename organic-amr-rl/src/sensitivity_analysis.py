"""
Sensitivity and Hyperparameter Analysis for Q-Learning Agent
Author: Hayden Hedman
Revised: 2026-03

Purpose:
Evaluate robustness of Q-learning agent performance across
learning rate (alpha), discount factor (gamma), and exploration rate (epsilon).

Outputs:
- qlearner_sensitivity_results.csv
- qlearner_sensitivity_ranked.csv
"""

import os
import time
import numpy as np
import pandas as pd

from copper_resistance_env import CopperResistanceEnv
from copper_q_learning_agent import QLearner
from rule_based_agent import RuleBasedPolicy
from random_policy import RandomPolicy


# ==============================================================
# CONFIGURATION
# ==============================================================

RANDOM_SEED = 42
EPISODES = 150          # Increased from original 100 for robustness
MAX_CYCLES = 40

HYPERPARAM_GRID = [
    {"alpha": 0.05, "gamma": 0.95, "epsilon": 0.10},
    {"alpha": 0.10, "gamma": 0.95, "epsilon": 0.10},  # baseline
    {"alpha": 0.20, "gamma": 0.95, "epsilon": 0.10},
    {"alpha": 0.10, "gamma": 0.90, "epsilon": 0.10},
    {"alpha": 0.10, "gamma": 0.95, "epsilon": 0.05},
    {"alpha": 0.10, "gamma": 0.95, "epsilon": 0.20},
]


# ==============================================================
# METRIC FUNCTIONS
# ==============================================================

def compute_auc(values):
    return np.trapz(values)


def run_agent_config(config):
    np.random.seed(RANDOM_SEED)

    env = CopperResistanceEnv(max_cycles=MAX_CYCLES)

    agent = QLearner(
        env,
        learning_rate=config["alpha"],
        discount=config["gamma"],
        epsilon=config["epsilon"]
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
        "alpha": config["alpha"],
        "gamma": config["gamma"],
        "epsilon": config["epsilon"],
        "AUC_MIC_Chloro_mean": np.mean(auc_chloro),
        "AUC_MIC_Chloro_sd": np.std(auc_chloro),
        "AUC_MIC_PolyB_mean": np.mean(auc_poly),
        "AUC_MIC_PolyB_sd": np.std(auc_poly),
        "Copper_Burden_mean": np.mean(copper_totals),
        "Copper_Burden_sd": np.std(copper_totals),
    }


# ==============================================================
# MAIN SENSITIVITY LOOP
# ==============================================================

def run_sensitivity():

    start_time = time.time()

    print("\nStarting Q-Learner Sensitivity Analysis...\n")

    results = []

    for config in HYPERPARAM_GRID:
        print(f"Running config: {config}")
        metrics = run_agent_config(config)
        results.append(metrics)

    
    #-----------------------
    summary_df = pd.DataFrame(results)

    # Rename columns for publication-ready output
    summary_df = summary_df.rename(columns={
        "alpha": "Learning Rate (α)",
        "gamma": "Discount Factor (γ)",
        "epsilon": "Exploration Rate (ε)",
        "AUC_MIC_Chloro_mean": "AUC MIC (Chloramphenicol), Mean",
        "AUC_MIC_Chloro_sd": "AUC MIC (Chloramphenicol), SD",
        "AUC_MIC_PolyB_mean": "AUC MIC (Polymyxin B), Mean",
        "AUC_MIC_PolyB_sd": "AUC MIC (Polymyxin B), SD",
        "Copper_Burden_mean": "Copper Burden, Mean",
        "Copper_Burden_sd": "Copper Burden, SD"
    })

    # Rank by lowest chloramphenicol AUC
    ranked_df = summary_df.sort_values(
        by="AUC MIC (Chloramphenicol), Mean",
        ascending=True
    )

    # ----------------------------------------------------------
    # Dynamic Output Directory Handling
    # ----------------------------------------------------------

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(src_dir, "outputs", "summary_tables")

    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(
        output_dir,
        "qlearner_sensitivity_results.csv"
    )

    ranked_path = os.path.join(
        output_dir,
        "qlearner_sensitivity_ranked.csv"
    )

    summary_df.to_csv(summary_path, index=False)
    ranked_df.to_csv(ranked_path, index=False)

    runtime = time.time() - start_time

    print("\nSensitivity analysis complete.")
    print(f"Results saved to: {output_dir}")
    print(f"Total runtime: {runtime:.2f} seconds\n")


# ==============================================================
# EXECUTION ENTRY POINT
# ==============================================================

if __name__ == "__main__":
    run_sensitivity()