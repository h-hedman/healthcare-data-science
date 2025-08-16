# CMS Medicare Data - SQL Examples using Synthetic SynPUF Dataset

# Load required libraries
library(DBI)
library(RSQLite)
library(tidyverse)
library(googlesheets4)

#-----------------------------------------------------------------------------------------------------------------------------------
# 1. Load & Prepare Data 
#-----------------------------------------------------------------------------------------------------------------------------------
# Load synthetic CMS data from Google Sheets
cms_gs <- read_sheet("https://docs.google.com/spreadsheets/d/18JDU4ghGCpRgUSDmrjV3LGqaU2HwGiJuejS-8kAZg0Q/edit?usp=sharing")

# Load state code reference
state_codes <- read_sheet("https://docs.google.com/spreadsheets/d/1lFTd1IgvkfYOVZGN-X-Mh_WTT0-9ewJPRe6uBDRM7ng/edit?usp=sharing")

# Simulate dummy education levels (for demo purposes only)
set.seed(123)
cms_gs <- cms_gs %>%
  mutate(
    EDUCATION_LEVEL = sample(1:3, size = n(), replace = TRUE),
    BENE_RACE_CD = recode(BENE_RACE_CD,
                          `1` = "WHITE",
                          `2` = "BLACK",
                          `3` = "OTHERS",
                          `5` = "HISPANIC"),
    BENE_SEX_IDENT_CD = recode(BENE_SEX_IDENT_CD,
                               `1` = "MALE",
                               `2` = "FEMALE")
  )

# Join with state codes
cms_df <- cms_gs %>%
  left_join(state_codes, by = "SP_STATE_CODE")
#-----------------------------------------------------------------------------------------------------------------------------------
# 2. Create SQLite DB and Load Data 
#-----------------------------------------------------------------------------------------------------------------------------------
# Connect to SQLite DB
con_cms <- dbConnect(SQLite(), dbname = "cms_sql.sqlite")

# Copy CMS dataframe into SQL table
copy_to(con_cms, cms_df, name = "cms_df", overwrite = TRUE)
#-----------------------------------------------------------------------------------------------------------------------------------
# 3. Run SQL Queries 
#-----------------------------------------------------------------------------------------------------------------------------------

# 3.1: Diabetes by Sex and Race
query1 <- dbGetQuery(con_cms, "
  SELECT 
    BENE_SEX_IDENT_CD AS sex,
    BENE_RACE_CD AS race,
    COUNT(DESYNPUF_ID) AS num_patients
  FROM cms_df
  WHERE SP_DIABETES = 1
  GROUP BY sex, race
  ORDER BY num_patients DESC;
")
print(query1)

# 3.2: Depression by State
query2 <- dbGetQuery(con_cms, "
  SELECT 
    state_code,
    COUNT(DESYNPUF_ID) AS num_patients
  FROM cms_df
  WHERE SP_DEPRESSN = 1
  GROUP BY state_code
  ORDER BY num_patients DESC;
")
print(query2)

# 3.3: Cancer Patients – Avg Reimbursement and Education Level
query3 <- dbGetQuery(con_cms, "
  SELECT 
    ROUND(AVG(PPPYMT_IP), 2) AS avg_inpatient_reimb,
    ROUND(AVG(MEDREIMB_OP), 2) AS avg_outpatient_reimb,
    ROUND(AVG(EDUCATION_LEVEL), 2) AS avg_education_level
  FROM cms_df
  WHERE SP_CNCR = 1;
")
print(query3)
#-----------------------------------------------------------------------------------------------------------------------------------
#  4. Clean up 
#-----------------------------------------------------------------------------------------------------------------------------------
# Disconnect from the database
dbDisconnect(con_cms)
