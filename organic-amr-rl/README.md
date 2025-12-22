# Reinforcement Learning Simulation for Copper-Induced Antimicrobial Resistance

**Preprint:** https://www.biorxiv.org/content/10.64898/2025.12.18.695270v1

**Manuscript under review:** *Applied Microbiology (MDPI)*

**Foundational reference for copper exposure assumptions:**  
Li, J., Phulpoto, I. A., Zhang, G., & Yu, Z. (2021). *Acceleration of emergence of E. coli antibiotic resistance in a simulated sublethal concentration of copper and tetracycline co-contaminated environment*. **AMB Express**, 11(1), 14.  
DOI: 10.1186/s13568-020-01173-6

This repository contains the simulation code used to evaluate reinforcement learning and baseline control strategies for managing copper exposure and antimicrobial resistance dynamics in a simplified microbial environment. Copper concentration ranges and resistance behavior are informed by prior experimental findings, including Li et al. (2021), and adapted for a simulation-based reinforcement learning framework.

---

## Project Contents (src)

All scripts are located in the repository root for transparency and reproducibility.

- **copper_resistance_env.py**  
  Defines the custom simulation environment modeling microbial growth, copper exposure, and resistance dynamics.

- **copper_q_learning_agent.py**  
  Implements the Q-learning agent used to adaptively administer copper based on observed system states.

- **random_policy.py**  
  Baseline agent that selects actions uniformly at random.

- **rule_based_agent.py**  
  Deterministic baseline agent using fixed heuristic decision rules.

- **run_experiments.py**  
  Main orchestration script for running simulations across agents, seeds, and experimental conditions.

- **generate_combined_cycles.py**  
  Aggregates simulation cycles and harmonizes outputs across experimental runs.

- **generate_tables.py**  
  Produces summary tables used in the manuscript (performance metrics, resistance burden, growth inhibition).

- **generate_plots.py**  
  Generates figures visualizing learning behavior, resistance trajectories, and comparative performance.

- **qc_diagnostics.py**  
  Quality control and diagnostic checks for simulation stability and output validity.

---

## Notes

- Applied simulation framework for understanding antimicrobial resistance under organic compound exposure.  
- Scripts generate all experiments, figures, and tables reported in the manuscript.
