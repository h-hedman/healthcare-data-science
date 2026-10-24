# Usage from the repo root: Rscript scripts/publish-app.R app_sir
# Explicit IDs update existing content and avoid consuming another Free app slot.
source("scripts/project-library.R")
args <- commandArgs(trailingOnly = TRUE)
apps <- read.csv("deployment/apps.csv", stringsAsFactors = FALSE)
if (length(args) != 1L || !args[1] %in% apps$app) {
  stop("Pass exactly one app name: ", paste(apps$app, collapse = ", "))
}
app <- apps[apps$app == args[1], ]
records <- rsconnect::deployments(app$app_path, serverFilter = "connect.posit.cloud")
if (nrow(records) != 1L || as.character(records$appId) != app$content_id) {
  stop("Run scripts/migrate-deployment-records.R first; deployment destination is not verified.")
}
rsconnect::deployApp(
  appDir = app$app_path,
  appFiles = "app.R",
  appPrimaryDoc = "app.R",
  appId = app$content_id,
  appName = app$app,
  account = "healthify-flu",
  server = "connect.posit.cloud",
  launch.browser = FALSE
)
