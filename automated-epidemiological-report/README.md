# Automated Epidemiological Report Tool

A lightweight epidemiological reporting system originally developed during a CDC emergency response. This tool automates the generation of clean, structured PDF reports from raw surveillance data, designed for use in low-resource and disconnected environments.

## Overview

This project streamlines outbreak reporting by parsing raw line-listed data to produce standardized epidemiological summaries. Outputs include tables and charts highlighting recent case trends, demographics, and condition-level alerts. Originally deployed in field settings with limited internet, the tool was later scaled for broader use.

## Key Features

- Generates professionally formatted PDF reports with minimal setup
- Designed for environments without reliable internet access
- Processes daily case data to summarize:
  - 7-day and 28-day caseload trends
  - Key disease metrics by condition and demographic
  - Condition-specific alert levels (e.g., high, medium, low)
- Includes optional Shiny front-end for rapid exploration or QA

## File Structure

- `app_automated_epi_report.R` – Optional Shiny app (interactive frontend)
- `epi_report_dummy_data_generator.R` – Script to generate synthetic case data
- `demo_infectious_cases.csv` – Example input data (line list format)
- `report_<timestamp>.pdf` – Output report with tables, plots, and summaries

## Example Use Case

During a CDC-led emergency response, this tool was used to automate daily infectious disease reports across refugee camp settings. It replaced hours of manual formatting and supported timely decision-making in the field.

## How to Use

1. Clone the repo and install required R packages (`tidyverse`, `rmarkdown`, etc.)
2. Supply a dataset in line-listed format or use the demo file
3. Run the script or app:
   ```r
   source("app_automated_epi_report.R")
