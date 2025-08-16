# ----------------------------------------------------------------------------------------------------------------
# AMR Exposure Modeling Script
# Author: Hayden Hedman
# Date: 2018-10-21
# ----------------------------------------------------------------------------------------------------------------
#  Load packages with helper check
if (!requireNamespace("pacman", quietly = TRUE)) install.packages("pacman")
pacman::p_load(lme4, lmerTest, gee, tidyr)
# ----------------------------------------------------------------------------------------------------------------
# Simulate Dummy Data 
# ----------------------------------------------------------------------------------------------------------------
set.seed(64)
num_samples <- 4000
community_names <- c("Caelid", "Raccoon_City")

amr_df <- data.frame(
  sample_number = sample(1:4, num_samples, replace = TRUE),
  sample_period = sample(1:4, num_samples, replace = TRUE),
  communtiy = sample(community_names, num_samples, replace = TRUE),
  distance_nearest_household_km = runif(num_samples, min = 0.2, max = 3.0),
  species = rbinom(num_samples, size = 1, prob = 0.8),  # 1 = human, 0 = animal
  AMP = rbinom(num_samples, 1, 0.5),
  AMC = rbinom(num_samples, 1, 0.5),
  CTX = rbinom(num_samples, 1, 0.5),
  CF  = rbinom(num_samples, 1, 0.5),
  C   = rbinom(num_samples, 1, 0.5),
  CIP = rbinom(num_samples, 1, 0.5),
  SXT = rbinom(num_samples, 1, 0.5),
  S   = rbinom(num_samples, 1, 0.5),
  GM  = rbinom(num_samples, 1, 0.5),
  TE  = rbinom(num_samples, 1, 0.5),
  G   = rbinom(num_samples, 1, 0.5),
  ENO = rbinom(num_samples, 1, 0.5)
)
# ----------------------------------------------------------------------------------------------------------------
#  Assign Unique IDs 
# ----------------------------------------------------------------------------------------------------------------
amr_df$id <- as.integer(factor(with(amr_df, paste(communtiy, species, sample_period))))
# ----------------------------------------------------------------------------------------------------------------
#  Create Dummy Period Indicators 
# ----------------------------------------------------------------------------------------------------------------
amr_df <- amr_df %>%
  mutate(
    sample_period_1 = ifelse(sample_period == 1, 1, 0),
    sample_period_2 = ifelse(sample_period == 2, 1, 0),
    sample_period_3 = ifelse(sample_period == 3, 1, 0)
  )
# ----------------------------------------------------------------------------------------------------------------
#  Reshape to Long Format 
# ----------------------------------------------------------------------------------------------------------------
amr_long <- gather(amr_df, key = "ab", value = "amr", AMP:ENO)
# ----------------------------------------------------------------------------------------------------------------
#  Mixed Effects Model: Spatial Exposure 
# ----------------------------------------------------------------------------------------------------------------
glmm_space_func <- function(antibiotic) {
  s_drug <- subset(amr_long, ab == antibiotic)
  mf1 <- formula(amr ~ distance_nearest_household_km + (1 | sample_period) + (1 | communtiy) + (1 | id))
  mem1 <- glmer(mf1, data = s_drug, family = binomial)
  cc <- coef(summary(mem1))
  colnames(cc)[2] <- "SE"
  
  citab <- with(as.data.frame(cc), cbind(
    lwr = Estimate - 1.96 * SE,
    upr = Estimate + 1.96 * SE
  ))
  
  rownames(citab) <- rownames(cc)
  return(cbind(cc, citab))
}
# ----------------------------------------------------------------------------------------------------------------
# Example: spatial effect for CTX
glmm_space_func("CTX")
# ----------------------------------------------------------------------------------------------------------------
#  Mixed Effects Model: Temporal Trends 
# ----------------------------------------------------------------------------------------------------------------
glmm_time_func <- function(antibiotic) {
  s_drug <- subset(amr_long, ab == antibiotic)
  
  # Add temporal indicators (1/0)
  s_drug <- s_drug %>%
    mutate(
      period_2 = ifelse(sample_period == 2, 1, 0),
      period_3 = ifelse(sample_period == 3, 1, 0)
    )
  
  mf2 <- formula(amr ~ period_2 + period_3 + (1 | communtiy) + (1 | id))
  mem2 <- glmer(mf2, data = s_drug, family = binomial)
  cc <- coef(summary(mem2))
  colnames(cc)[2] <- "SE"
  
  citab <- with(as.data.frame(cc), cbind(
    lwr = Estimate - 1.96 * SE,
    upr = Estimate + 1.96 * SE
  ))
  
  rownames(citab) <- rownames(cc)
  return(cbind(cc, citab))
}
# ----------------------------------------------------------------------------------------------------------------
# Example: temporal change for AMP
glmm_time_func("AMP")
# ----------------------------------------------------------------------------------------------------------------

