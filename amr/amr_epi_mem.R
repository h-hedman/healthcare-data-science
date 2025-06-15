#title: "amr_epi"
#author: Hayden Hedman
#date: "2018-10-21" ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Load packages -------------------------------------------------------------------------------------------------------------------------------------------------------------------
install.packages(setdiff("pacman", rownames(installed.packages()))) 
## PACKAGE PACMAN LOADS AND CHECKS IF EACH PACKAGE NEEDS INSTALLED
pacman::p_load(lme4,tidyr, lmerTest, gee)

# Create Dummy Data (remove seed to randomize results) ----------------------------------------------------------------------------------------------------------------------------
set.seed(1)
# Variables -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
num_samples = 4000
community_names = c("Caelid","Raccoon_City")
# Create simulated per pathogen ---------------------------------------------------------------------------------------------------------------------------------------------------
amr_df <- data.frame(sample_number                    = sample(1:4, num_samples, replace=T, prob=c(0.25, 0.25, 0.25, 0.25)),
                     sample_period                    = sample(1:4, num_samples, replace=T, prob=c(0.25, 0.25, 0.25, 0.25)),
                     communtiy                        = sample(community_names, num_samples, replace=T, prob=c(0.5, 0.5)),
                     distance_nearest_household_km    = runif(num_samples, min = 0.2, max = 3.0),
                     species                          = rbinom(n=num_samples, size=1, prob=0.8), # 1 = human, 0 = animal
                     AMP                              = rbinom(n=num_samples, size=1, prob=0.5),
                     AMC                              = rbinom(n=num_samples, size=1, prob=0.5),
                     CTX                              = rbinom(n=num_samples, size=1, prob=0.5),
                     CF                               = rbinom(n=num_samples, size=1, prob=0.5),
                     C                                = rbinom(n=num_samples, size=1, prob=0.5),
                     CIP                              = rbinom(n=num_samples, size=1, prob=0.5),
                     SXT                              = rbinom(n=num_samples, size=1, prob=0.5),
                     S                                = rbinom(n=num_samples, size=1, prob=0.5),
                     GM                               = rbinom(n=num_samples, size=1, prob=0.5),
                     TE                               = rbinom(n=num_samples, size=1, prob=0.5),
                     G                                = rbinom(n=num_samples, size=1, prob=0.5),
                     ENO                              = rbinom(n=num_samples, size=1, prob=0.5))

# Create unique id ----------------------------------------------------------------------------------------------------------------------------------------------------------------
amr_df$id <- as.integer(factor(with(amr_df, paste(communtiy, species, sample_period))))
# Categorical sample periods ------------------------------------------------------------------------------------------------------------------------------------------------------
amr_df$sample_period_1 = 0 
amr_df$sample_period_1[which(amr_df$sample_period == 1)] <- "1"
amr_df$sample_period_2 = 0 
amr_df$sample_period_2[which(amr_df$sample_period == 2)] <- "1"
amr_df$sample_period_3 = 0 
amr_df$sample_period_3[which(amr_df$sample_period == 3)] <- "1"
# Transform data long to wide for functions ---------------------------------------------------------------------------------------------------------------------------------------
amr_long = gather(amr_df, key="ab", value="amr", 6:17)
# Mixed Effects Models ------------------------------------------------------------------------------------------------------------------------------------------------------------
# Distance to nearest exposure of intensive farming poultry -----------------------------------------------------------------------------------------------------------------------
# Write function for each model
glmm_space_func <- function(antibiotic) {
  s_drug <- subset(amr_long, ab==antibiotic)
  mf1 <- formula(amr~distance_nearest_household_km + (1|sample_period) + (1|communtiy) + (1|id)) 
  mem1 <- glmer(mf1, data=s_drug, family=binomial)
  cc <- coef(summary(mem1))
  colnames(cc)[2] <- "SE"
  
  citab <- with(as.data.frame(cc),
                cbind(lwr=Estimate-1.96*SE,
                      upr=Estimate+1.96*SE))
  rownames(citab) <- rownames(cc)
  cbind(cc,citab)
}

# Example AMR spatial exposure function
glmm_space_func("CTX")

# Change in AMR levels over time  --------------------------------------------------------------------------------------------------------------------------------------------------
glmm_time_func <- function(antibiotic) {
  s_drug <- subset(amr_long, ab==antibiotic)
  mf2 <- formula(amr~period_2 + period_3+(1|community)+(1|id)) 
  mem2 <- glmer(mf1, data=s_drug, family=binomial)
  cc <- coef(summary(mem2))
  colnames(cc)[2] <- "SE"
  
  citab <- with(as.data.frame(cc),
                cbind(lwr=Estimate-1.96*SE,
                      upr=Estimate+1.96*SE))
  rownames(citab) <- rownames(cc)
  cbind(cc,citab)
}

# Example model
glmm_time_func("AMP")

# End of script ----------------------------------------------------------------------------------------------------------------------------------------------------------------------

