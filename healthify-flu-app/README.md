# **Healthify: Influenza Population Health Research Metrics Web App**

## Overview

**Healthify** is a data-driven web application designed for **population health researchers** to explore and visualize trends in **seasonal influenza hospitalization**. The app integrates public data from the [CDC's FluSurv-NET](https://gis.cdc.gov/GRASP/Fluview/FluHospRates.html) and provides analytical tools to evaluate flu dynamics across space, time, and demographics.

This project originated as a **pilot tool I led the development of** during a practicum for *CS-6440: Introduction to Health Informatics* at Georgia Tech. It has since been extended and refined to better serve public health professionals and epidemiological researchers.

The platform combines **epidemiological, biostatistical, and machine learning methods** to offer insights on:

- Geographic and demographic trends in flu hospitalization  
- Impact of social vulnerability on epidemic curves  
- Predictive modeling of disease burden  

---

## Key Features

### 🔹 Data Collection & Processing

- **Source:** [CDC FluSurv-NET](https://gis.cdc.gov/GRASP/Fluview/FluHospRates.html)  
- **Coverage:** ~30M individuals across 90 counties in 14 states (~9% of U.S. population)  
- **Workflow:**  
  - Data ingestion and preprocessing in **R**  
  - Stratified summaries by **age, race, time period**  
  - Integration of simulation-ready data formats  

### 🔹 Analytics & Modeling

- **Descriptive Statistics:** Hospitalization incidence and demographic breakdowns  
- **Predictive Modeling:**
  - Logistic regression models for high/low incidence classification  
  - Time series forecasting (ARIMA, SARIMA)  
  - Compartmental models (SIR, SEIR) for simulating outbreaks  
- **Geospatial Mapping:** County- and state-level flu burden visualizations  

### 🔹 End Users

This tool is designed for researchers and organizations involved in **population-based health outcomes research**, including:

- **Public health departments** (e.g., University of Michigan, Emory University)  
- **Government institutions** (e.g., CDC, NIH)  
- **NGOs and foundations** (e.g., Kaiser Permanente)  
- **Biopharma and biotech companies** (e.g., Merck, Pfizer, Eli Lilly)  

### 🔹 Data Visualization

Due to free-tier hosting limitations, a hybrid deployment approach is used:

- **R Shiny**: For interactive dashboards  
- **DataWrapper**: For fast, embeddable static charts  

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.gatech.edu/hhedman3/cs-6440-practicum-healthify.git
cd cs-6440-practicum-healthify
```

### 2. Run Locally

Open the Shiny app files (`ui.R`, `server.R`, etc.) using RStudio and click **Run App**. You may need the following packages:

```r
install.packages(c("shiny", "tidyverse", "lubridate", "ggplot2", "data.table"))
```

---

## License

This repository is intended for educational and demonstrative purposes. All CDC data is publicly available. Please cite appropriately when using or extending this tool.
