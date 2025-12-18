# =============================================================================
# GLMM Analysis for MIC Trajectories
# Author: Hayden Hedman
# =============================================================================

library(lme4)
library(lmerTest)
library(dplyr)
library(readr)
library(emmeans)

# Set global options for lmerTest and pbkrtest to handle large datasets
options(pbkrtest.limit = 8000)  # For PBKRTest package
options(lmerTest.limit = 8000)  # For lmerTest package

# =============================================================================
# HARD-CODED PATHS — CHANGE FOR PUBLIC RELEASE
# =============================================================================
##raw_csv_dir <- "ADD local file path"
##summary_dir <- "add local file path"

# This is the file you already have (not in raw_csv, but in summary_tables)
combined_path <- file.path(summary_dir, "combined_agent_cycles.csv")

# Make sure directory exists
if (!dir.exists(summary_dir)) dir.create(summary_dir, recursive = TRUE)

# =============================================================================
# LOAD DATA
# =============================================================================
df <- read_csv(combined_path)

df$agent   <- factor(df$agent)
df$episode <- factor(df$episode)

# =============================================================================
# MODEL 1 — MIC (Chloramphenicol)
# =============================================================================
model_chloro <- lmer(MIC_chloro ~ cycle * agent + (1 | episode), data=df)

# Open a text file to capture all outputs for chloramphenicol
output_file_chloro <- file.path(summary_dir, "glmm_chloro_results.txt")
capture.output(
  cat("GLMM Analysis for Chloramphenicol:\n\n"),
  file = output_file_chloro
)

# Capture the GLMM summary (fixed effects, random effects, residuals)
chloro_summary <- summary(model_chloro)
capture.output(
  cat("GLMM Summary:\n"),
  chloro_summary,
  file = output_file_chloro,
  append = TRUE
)

# Capture the ANOVA table for chloramphenicol
chloro_anova <- anova(model_chloro)
capture.output(
  cat("\nANOVA Table (Type III):\n"),
  chloro_anova,
  file = output_file_chloro,
  append = TRUE
)

# Capture the 95% Confidence Intervals for the fixed effects
chloro_confint <- confint(model_chloro, level = 0.95)
capture.output(
  cat("\n95% Confidence Intervals for Fixed Effects:\n"),
  chloro_confint,
  file = output_file_chloro,
  append = TRUE
)

# =============================================================================
# MODEL 2 — MIC (Polymyxin B)
# =============================================================================
model_polyB <- lmer(MIC_polyB ~ cycle * agent + (1 | episode), data=df)

# Open a text file to capture all outputs for polymyxin B
output_file_polyB <- file.path(summary_dir, "glmm_polyB_results.txt")
capture.output(
  cat("GLMM Analysis for Polymyxin B:\n\n"),
  file = output_file_polyB
)

# Capture the GLMM summary (fixed effects, random effects, residuals)
polyB_summary <- summary(model_polyB)
capture.output(
  cat("GLMM Summary:\n"),
  polyB_summary,
  file = output_file_polyB,
  append = TRUE
)

# Capture the ANOVA table for polymyxin B
polyB_anova <- anova(model_polyB)
capture.output(
  cat("\nANOVA Table (Type III):\n"),
  polyB_anova,
  file = output_file_polyB,
  append = TRUE
)

# Capture the 95% Confidence Intervals for the fixed effects
polyB_confint <- confint(model_polyB, level = 0.95)
capture.output(
  cat("\n95% Confidence Intervals for Fixed Effects:\n"),
  polyB_confint,
  file = output_file_polyB,
  append = TRUE
)

# =============================================================================
# PAIRWISE CONTRASTS (OPTIONAL) with 95% CIs
# =============================================================================
emm_chloro <- emmeans(model_chloro, ~ agent)
chloro_pw <- contrast(emm_chloro, method="pairwise", adjust="bonferroni")

# Adding 95% CIs to pairwise contrasts output for chloramphenicol
chloro_pw_summary <- summary(chloro_pw, infer=c(TRUE, TRUE))
write.csv(
  as.data.frame(chloro_pw_summary),
  file=file.path(summary_dir, "glmm_chloro_pairwise.csv"),
  row.names=FALSE
)

emm_polyB <- emmeans(model_polyB, ~ agent)
polyB_pw <- contrast(emm_polyB, method="pairwise", adjust="bonferroni")

# Adding 95% CIs to pairwise contrasts output for polymyxin B
polyB_pw_summary <- summary(polyB_pw, infer=c(TRUE, TRUE))
write.csv(
  as.data.frame(polyB_pw_summary),
  file=file.path(summary_dir, "glmm_polyB_pairwise.csv"),
  row.names=FALSE
)
