# ---------------------------------------------------------------------------
# Author: Hayden Hedman
# Date: 2021-10-16
# Description: A Shiny app to generate automated epidemiological reports for
# infectious diseases with upload, visualizations, and PDF export.
# ---------------------------------------------------------------------------
# Creates dummy data for testing automated epi report (PDF) app
set.seed(888)
demo_data <- data.frame(
  patient_id = 1:1000,
  location = "Pallet Town",
  pathogen = sample(c("Influenza", "Measles", "TB", "Pertusis"), 1000, replace = TRUE),
  test_date = sample(seq(Sys.Date() - 60, Sys.Date(), by = "day"), 1000, replace = TRUE),
  test_result = rbinom(1000, size = 1, prob = 0.6)
)

# Save to CSV
write.csv(demo_data, "demo_infectious_cases.csv", row.names = FALSE)
# ---------------------------------------------------------------------------
