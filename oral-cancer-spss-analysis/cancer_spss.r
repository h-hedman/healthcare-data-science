# -----------------------------------------------------------------------------------------
# Cancer Data Analysis Using SPSS Dataset
# -----------------------------------------------------------------------------------------
# Load libraries
library(tidyverse)
library(haven)
library(ggplot2)
library(patchwork)
library(kableExtra)
# -----------------------------------------------------------------------------------------
# 1. Load and Prepare Data
# -----------------------------------------------------------------------------------------
# Source: Mid-Michigan Medical Center, 1999
# Data dictionary: http://calcnet.mth.cmich.edu/org/spss/prj_cancer_data.htm
cancer_df <- read_sav('cancer.sav')

# Subset treatment groups
plac <- subset(cancer_df, TRT == 0)
treatment <- subset(cancer_df, TRT == 1)
# -----------------------------------------------------------------------------------------
# 2. T-tests: Oral Condition Differences 
# -----------------------------------------------------------------------------------------
# Week 0: Initial stage
t.test_overall <- t.test(plac$TOTALCIN, treatment$TOTALCIN, var.equal = FALSE)

# Weeks 2, 4, 6
t.test_wk2 <- t.test(plac$TOTALCW2, treatment$TOTALCW2, var.equal = FALSE)
t.test_wk4 <- t.test(plac$TOTALCW4, treatment$TOTALCW4, var.equal = FALSE)
t.test_wk6 <- t.test(plac$TOTALCW6, treatment$TOTALCW6, var.equal = FALSE)

# Create summary table
ttest_df <- tibble(
  `Oral condition at week` = c("2", "4", "6"),
  `t-value` = c(t.test_wk2$statistic, t.test_wk4$statistic, t.test_wk6$statistic),
  `p-value` = c(t.test_wk2$p.value, t.test_wk4$p.value, t.test_wk6$p.value)
)

print(ttest_df)
# -----------------------------------------------------------------------------------------
# 3. Mean Plot by Week 
# -----------------------------------------------------------------------------------------
# Select and rename columns
plot_df <- cancer_df[, c("TRT", "TOTALCW2", "TOTALCW4", "TOTALCW6")]
colnames(plot_df)[1] <- "Group"

plot_df$Group <- factor(ifelse(plot_df$Group == 0, "Placebo", "Treatment"))
plot_df[is.na(plot_df)] <- 0

# Summarize means
sum_wk <- plot_df %>%
  group_by(Group) %>%
  summarize(
    mean_wk2 = mean(TOTALCW2),
    mean_wk4 = mean(TOTALCW4),
    mean_wk6 = mean(TOTALCW6)
  )

# Convert to long format for ggplot
data_long <- sum_wk %>%
  pivot_longer(cols = -Group, names_to = "week_number", values_to = "measurement") %>%
  mutate(week = case_when(
    week_number == "mean_wk2" ~ 2,
    week_number == "mean_wk4" ~ 4,
    week_number == "mean_wk6" ~ 6
  ))
# -----------------------------------------------------------------------------------------
# 4. Plots 
# -----------------------------------------------------------------------------------------
# Bar plot: Mean oral condition over time by group
mean_group_plot <- ggplot(data_long, aes(x = week, y = measurement, fill = Group)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  scale_fill_manual(values = c("Purple", "Yellow")) +
  labs(
    title = "Cancer Measurement Between Groups Over Time",
    x = "Week",
    y = "Mean Oral Condition"
  ) +
  theme_minimal()

# Scatter plot: Initial cancer condition by stage
scatter_plot_stage <- ggplot(cancer_df, aes(x = STAGE, y = TOTALCIN)) +
  geom_point() +
  labs(
    title = "Cancer Stage and Oral Condition at Initial Stage",
    x = "Stage",
    y = "Initial Oral Condition"
  ) +
  theme_minimal()

# Combine plots
mean_group_plot + scatter_plot_stage
# -----------------------------------------------------------------------------------------
