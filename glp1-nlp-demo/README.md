# GLP-1 NLP Demo Pipeline

This repository demonstrates a **lightweight, end-to-end NLP pipeline** for abstracting structured labels from unstructured clinical text, using GLP-1 medications as a motivating example.  
It is designed for **portfolio and interview demo purposes**, not production use.

---

## Project Goal

Simulate how healthcare NLP pipelines:
- Ingest heterogeneous clinical data
- Process unstructured notes
- Extract interpretable labels
- Store outputs in scalable, tabular formats suitable for analytics

---

## Directory Structure

glp1_nlp_demo/
├── data/
│ ├── clinical_notes.txt
│ ├── prescriptions.csv
│ ├── diagnoses_icd.csv
│ └── labels.csv
│
├── scripts/
│ ├── 00_generate_synthetic_glp1_data.py
│ └── 01_glp1_nlp_pipeline.ipynb



---

## Data Overview

All data are **synthetic** and generated solely for demonstration.

- **clinical_notes.txt**  
  Free-text clinical notes containing medication mentions, conditions, and temporal cues.

- **prescriptions.csv**  
  Simulated medication records (e.g., GLP-1 agents, start dates).

- **diagnoses_icd.csv**  
  Example ICD-coded diagnoses used for contextual grounding.

- **labels.csv**  
  Final structured outputs derived from the NLP pipeline.

---

## Scripts

### `00_generate_synthetic_glp1_data.py`
Generates realistic but fully synthetic:
- Clinical notes
- Prescription records
- Diagnosis tables

Used to ensure the pipeline is **reproducible and self-contained**.

---

### `01_glp1_nlp_pipeline.ipynb`
Implements the NLP pipeline:

1. Load structured + unstructured inputs  
2. Clean and normalize clinical text  
3. Identify GLP-1 medication mentions  
4. Resolve simple temporal/context rules  
5. Output interpretable labels to `labels.csv`

The notebook emphasizes **clarity and pipeline structure**, not model complexity.

---

## Key Design Notes

- Focuses on **label abstraction**, not LLM fine-tuning
- Prioritizes **auditability and interpretability**
- Mirrors real-world healthcare data constraints:
  - Messy text
  - Temporal ambiguity
  - Heterogeneous sources

---

## Disclaimer

This project uses **synthetic data only** and contains no PHI.  
It is intended solely as a **technical demonstration** of healthcare NLP workflows.

---

