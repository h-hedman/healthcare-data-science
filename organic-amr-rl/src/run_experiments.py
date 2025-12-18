# run_experiments.py
# -------------------------------------------------------------------------------------
"""
Run Experiments for Copper-Driven Antibiotic Resistance Study
Author: Hayden Hedman
Date: 2025-12-10

Evaluates three copper-control strategies:
1. Random Policy
2. Rule-Based Policy
3. Q-Learner

Outputs raw per-cycle CSV files into:  outputs/raw_csv/
"""
# -------------------------------------------------------------------------------------
# Dynamically configure Python path so /scripts works as a package
# -------------------------------------------------------------------------------------
import sys
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()       # .../src/scripts/run_experiments.py
SCRIPTS_DIR = CURRENT_FILE.parent             # .../src/scripts
PROJECT_ROOT = SCRIPTS_DIR.parent             # .../src/

# Allow "import scripts.*"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# load libraries and other scripts
# -------------------------------------------------------------------------------------
import numpy as np
import pandas as pd

from scripts.copper_resistance_env import CopperResistanceEnv
from scripts.random_policy import RandomPolicy
from scripts.rule_based_agent import RuleBasedPolicy
from scripts.copper_q_learning_agent import QLearner
# -------------------------------------------------------------------------------------
# Helper: run agent for N episodes
# -------------------------------------------------------------------------------------
def run_agent(env, agent, episodes=50, label="agent"):
    results = []

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        agent.reset()

        done = False
        total_reward = 0
        cycle_data = []

        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)

            # Q-learner update only if the agent defines learn()
            if hasattr(agent, "learn"):
                agent.learn(state, action, reward, next_state, done)

            total_reward += reward

            copper, mic_c, mic_p, cycle, growth = next_state

            cycle_data.append({
                "episode": ep,
                "cycle": cycle,
                "copper": copper,
                "MIC_chloro": mic_c,
                "MIC_polyB": mic_p,
                "growth_inhibition": growth,
                "reward": reward,
            })

            state = next_state

        results.extend(cycle_data)
        print(f"{label}: Completed episode {ep} with total reward {total_reward:.2f}")

    return results

# -------------------------------------------------------------------------------------
# Main experiment runner
# -------------------------------------------------------------------------------------
def main():

    # Output folder for raw CSVs
    output_dir = PROJECT_ROOT / "outputs" / "raw_csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing environment and agents...")
    env = CopperResistanceEnv(max_cycles=40)

    random_agent = RandomPolicy(env.action_space)
    rule_agent = RuleBasedPolicy()
    q_agent = QLearner(env, learning_rate=0.1, discount=0.95, epsilon=0.1)

    # ---------------- Random ----------------
    print("\nRunning Random Policy...")
    random_results = run_agent(env, random_agent, episodes=50, label="random")
    pd.DataFrame(random_results).to_csv(output_dir / "random_policy_results.csv", index=False)

    # ---------------- Rule-based ----------------
    print("\nRunning Rule-Based Policy...")
    rule_results = run_agent(env, rule_agent, episodes=50, label="rule_based")
    pd.DataFrame(rule_results).to_csv(output_dir / "rule_based_results.csv", index=False)

    # ---------------- Q-Learner ----------------
    print("\nRunning Q-Learner Policy...")
    q_results = run_agent(env, q_agent, episodes=100, label="q_learner")
    pd.DataFrame(q_results).to_csv(output_dir / "q_learner_results.csv", index=False)

    print("\nAll experiments completed successfully.")
    print(f"Raw CSV files saved to: {output_dir.resolve()}")

# -------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
# -------------------------------------------------------------------------------------
