### 1. Data Source and Coverage
- **Data Source**: [CDC FluSurv-NET](https://gis.cdc.gov/GRASP/Fluview/FluHospRates.html)
- **Coverage**: Influenza hospitalization rates from over **90 U.S. counties** across **14 states**, covering approximately **9% of the U.S. population**.

### 2. Data Processing Steps

**Step 1: Data Extraction**
- Download weekly data files in CSV format from the CDC FluSurv-NET portal.

**Step 2: Data Cleaning (Python & R)**
- Use **Python (Pandas)** for data cleaning (standardize formats, handle missing values, and correct inconsistencies).
- Standardize county names, state abbreviations, and date formats.

**Step 3: Data Transformation**
- Convert data from wide to long format using **Python (pandas.melt())** or **R (tidyr::pivot_longer())**.
- Aggregate data by state, county, age group, and demographics.
- Calculate hospitalization rates per 100,000 population.

**Step 4: Statistical Summarization**
- Calculate weekly incidence rates by county, state, and nationally.
- Use rolling averages to smooth data trends.
- Perform exploratory analysis with visualization libraries in **R (ggplot2)** and **Python (matplotlib, seaborn)**.

**Step 5: Final Dataset**
- Export cleaned and aggregated datasets as CSV files for analysis and visualization.
