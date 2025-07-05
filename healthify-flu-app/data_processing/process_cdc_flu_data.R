# Author: Healthify
# Date: 2020-03-30
# Description: This script processes the CDC flu data for all other apps and models
# ---------------------------------------------------------------------------------
# Load libraries
require(tidyverse)
require(dplyr)
require(ggplot2)
require(lubridate)
require(tidyr)
require(readr)
# ---------------------------------------------------------------------------------
# Set working directory
#setwd(choose.dir())
setwd("G:/My Drive/Projects/OMSCS/Courses/CS 6440 - Introduction to Health Informatics/Assignments/IHI - Practicum Project/data_processing")
# ---------------------------------------------------------------------------------
# Load data
df_raw <- read.csv("FluSurveillance_Custom_Download_Data_RAW.csv")

# Remove all the junk rows
df <- subset(df_raw, !is.na(YEAR.1))
df<- subset(df, CUMULATIVE.RATE != "null")


# Clean up cumulative rate column
df <- df[!is.na(df$CUMULATIVE.RATE), ]

df$cumulative_rate <- as.numeric(df$CUMULATIVE.RATE)

str(df$CUMULATIVE.RATE)
class(df$CUMULATIVE.RATE)
unique(df$CUMULATIVE.RATE[!grepl("^-?[0-9.]+$", as.character(df$CUMULATIVE.RATE))])

df$CUMULATIVE.RATE <- trimws(df$CUMULATIVE.RATE)  # Remove leading/trailing spaces
df$CUMULATIVE.RATE <- gsub("[^0-9.-]", "", df$CUMULATIVE.RATE)  # Remove any non-numeric characters
df$CUMULATIVE.RATE <- as.numeric(df$CUMULATIVE.RATE)

str(df$CUMULATIVE.RATE)
class(df$CUMULATIVE.RATE)
# ---------------------------------------------------------------------------------
# Clean Age Categories
df <- df %>%
  mutate(
    clean_age_category = case_when(
      AGE.CATEGORY %in% c("<18", "0-<1 yr", "0-4 yr", "1-4 yr", "5-11 yr", "5-17 yr", "12-17 yr") ~ "0-17",
      AGE.CATEGORY %in% c("18-29 yr", "18-49 yr", "30-39 yr", "40-49 yr") ~ "18-49",
      AGE.CATEGORY == "50-64 yr" ~ "50-64",
      AGE.CATEGORY %in% c("65-74 yr", ">=65 yr") ~ "65-74",
      AGE.CATEGORY %in% c("75-84 yr", ">=75 yr") ~ "75-84",
      AGE.CATEGORY == ">=85 yr" ~ "85+",
      TRUE ~ NA_character_  # Assign NA if an unexpected value appears
    )
  )
# Check for strings in column
#unique(df$CUMULATIVE.RATE[is.na(as.numeric(as.character(df$CUMULATIVE.RATE)))])
# ---------------------------------------------------------------------------------
# (1) Process data for SUMMARY CASE DATA

#Summarize data for OVERALL - summary case data
en_df <- subset(df, NETWORK == "IHSP" & 
                         AGE.CATEGORY == "Overall" & 
                         SEX.CATEGORY == "Overall" & 
                         RACE.CATEGORY == "Overall" &
                         VIRUS.TYPE.CATEGORY == "Overall")

# Convert CUMULATIVE.RATE to numeric (handling potential conversion issues)


# QC check on data type
en_df <- en_df %>%
  mutate(CUMULATIVE.RATE = as.numeric(CUMULATIVE.RATE))

# Filter and summarize
en_df2 <- en_df %>%
  filter(NETWORK == "IHSP",
         AGE.CATEGORY == "Overall",
         SEX.CATEGORY == "Overall",
         RACE.CATEGORY == "Overall",
         VIRUS.TYPE.CATEGORY == "Overall") %>%
  group_by(YEAR.1) %>%
  summarise(
    median_rate = median(CUMULATIVE.RATE, na.rm = TRUE),
    min_rate = min(CUMULATIVE.RATE, na.rm = TRUE),
    max_rate = max(CUMULATIVE.RATE, na.rm = TRUE)
  )

# Clean up table names before print
clean_names_overall <- c("Year", "Median Rate", "Min Rate", "Max Rate")
en_df2_final <- en_df2
names(en_df2_final) <- clean_names_overall

# CHECK OUTPUT
#print(en_df2)
write.csv(en_df2_final, "overall_summary_incidence_2009_2025.csv",row.names=F)
# ---------------------------------------------------------------------------------
# (2) Process data for DEMOGRAPHICS
# ---------------------------------------------------------------------------------
# Filter: 
## Albany NY
## Entire Network
# Change: "New York - Rochester " to New York
demo_df <- subset(df, CATCHMENT != "New York - Albany" & 
                   CATCHMENT != "Entire Network")

# New York - Albany
demo_df$CATCHMENT[which(demo_df$CATCHMENT == "New York - Rochester")] <- "New York"
# ---------------------------------------------------------------------------------
# (2A) Process data for DEMOGRAPHICS - AGE GROUPS
age_df <- subset(demo_df,  AGE.CATEGORY != "Overall")

#str(age_df$CUMULATIVE.RATE)
#class(age_df$CUMULATIVE.RATE)

# Summarize by age group
summ_age_wide <- age_df %>%
  filter(!is.na(clean_age_category)) %>%  # Exclude NA groups before summarizing
  group_by(YEAR.1, clean_age_category) %>%
  summarise(
    median_rate = median(CUMULATIVE.RATE, na.rm = TRUE),
    .groups = "drop"  # Suppress grouping message
  ) %>%
  pivot_wider(
    names_from = clean_age_category, 
    values_from = median_rate
  )


# Clean up table names before print
colnames(summ_age_wide)[1]<-"Year"


write.csv(summ_age_wide, "age_group_summary_incidence_2009_2025.csv",row.names=F)
# ---------------------------------------------------------------------------------
# (2B) Subset Race/Ethnicity

race_df <- subset(demo_df,  RACE.CATEGORY != "Overall")

# Summarize by age group
summ_race_wide <- race_df %>%
  filter(!is.na(RACE.CATEGORY)) %>%  # Exclude NA values in race category
  group_by(YEAR.1, RACE.CATEGORY) %>%
  summarise(
    median_rate = median(CUMULATIVE.RATE, na.rm = TRUE),
    .groups = "drop"  # Suppress grouping message
  ) %>%
  pivot_wider(
    names_from = RACE.CATEGORY, 
    values_from = median_rate
  )

# Clean up table names before print
colnames(summ_race_wide)[1]<-"Year"

write.csv(summ_race_wide, "race_ethnicity_incidence_2009_2025.csv",row.names=F)
# ---------------------------------------------------------------------------------
# (2C) Subset by gender

sex_df <- subset(df,  SEX.CATEGORY != "Overall")


# Summarize by age group
summ_sex_wide <- sex_df %>%
  filter(!is.na(SEX.CATEGORY)) %>%  # Exclude NA values in race category
  group_by(YEAR.1, SEX.CATEGORY) %>%
  summarise(
    median_rate = median(CUMULATIVE.RATE, na.rm = TRUE),
    .groups = "drop"  # Suppress grouping message
  ) %>%
  pivot_wider(
    names_from = SEX.CATEGORY, 
    values_from = median_rate
  )

# Clean up table names before print
colnames(summ_sex_wide)[1]<-"Year"
write.csv(summ_sex_wide, "sex_incidence_data.csv", row.names=F)
# ---------------------------------------------------------------------------------
# (3) Geographic data


# Define state name to abbreviation mapping
state_abbrev_map <- c(
  "California" = "CA",
  "Colorado" = "CO",
  "Connecticut" = "CT",
  "Entire Network" = NA,  # Assign NA since it's not a state
  "Georgia" = "GA",
  "Idaho" = "ID",
  "Iowa" = "IA",
  "Maryland" = "MD",
  "Michigan" = "MI",
  "Minnesota" = "MN",
  "New Mexico" = "NM",
  "New York - Albany" = "NY",
  "New York - Rochester" = "NY",
  "North Carolina" = "NC",
  "Ohio" = "OH",
  "Oklahoma" = "OK",
  "Oregon" = "OR",
  "Rhode Island" = "RI",
  "South Dakota" = "SD",
  "Tennessee" = "TN",
  "Utah" = "UT"
)


# List of all 50 U.S. state abbreviations (excluding DC and territories)
all_states <- c(
  "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", 
  "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
  "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", 
  "VA", "WA", "WV", "WI", "WY"
)

# Get all years from the dataset
all_years <- unique(demo_df$YEAR.1)

# Expand grid to create all (YEAR, STATE) combinations **before summarization**
expanded_geo_df <- expand.grid(YEAR.1 = all_years, state_abbrev = all_states) %>%
  left_join(demo_df %>%
              mutate(state_abbrev = state_abbrev_map[CATCHMENT]) %>%
              filter(!is.na(state_abbrev)), 
            by = c("YEAR.1", "state_abbrev")) %>%
  mutate(CUMULATIVE.RATE = ifelse(is.na(CUMULATIVE.RATE), 0, CUMULATIVE.RATE))  # Assign missing states a 0


geo_df <- expanded_geo_df %>%
  group_by(YEAR.1, state_abbrev) %>%
  summarise(
    median_incidence = median(CUMULATIVE.RATE, na.rm = TRUE),
    .groups = "drop"
  )

# Clean up table names before print
colnames(geo_df)[1]<-"Year"
colnames(geo_df)[2]<-"State"
colnames(geo_df)[3]<-"Median Incidence"
# Write csv
write.csv(geo_df, "geographic_incidence_data.csv", row.names=F)
# --------------------------------------------------------------------------------- 
# (4) Process data for ML classification model

# Drop unnecessary columns
df_ml_race <- subset(df, RACE.CATEGORY != "Overall" & CATCHMENT != "Entire Network")
df_ml_age <- subset(df, AGE.CATEGORY != "Overall" & CATCHMENT != "Entire Network")
# ----------------------------------------------------------------------------------------
# Define threshold for high/low incidence in race subset
race_threshold <- median(df_ml_race$CUMULATIVE.RATE, na.rm = TRUE)
df_ml_race <- df_ml_race %>%
  mutate(high_incidence = ifelse(CUMULATIVE.RATE >= race_threshold, 1, 0))

# Write CSV for ML FLU Race data
write.csv(df_ml_race, "cleaned_flu_ml_race.csv", row.names=F)
# ----------------------------------------------------------------------------------------
# --- Subset for Age-Based Analysis ---
df_ml_age <- df %>%
  filter(AGE.CATEGORY != "Overall", CATCHMENT != "Entire Network") %>%
  select(YEAR.1, WEEK, CATCHMENT, clean_age_category, CUMULATIVE.RATE)

# Define threshold for high/low incidence in age subset
age_threshold <- median(df_ml_age$CUMULATIVE.RATE, na.rm = TRUE)
df_ml_age <- df_ml_age %>%
  mutate(high_incidence = ifelse(CUMULATIVE.RATE >= age_threshold, 1, 0))

# Save cleaned age dataset
write.csv(df_ml_age, "cleaned_flu_ml_age.csv", row.names=F)
# ----------------------------------------------------------------------------------------






















