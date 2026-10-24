# Run from the repository root after authenticating with rsconnect::connectCloudUser().
# This updates local publishing destinations; it does not publish any app.
source("scripts/project-library.R")
if (!requireNamespace("rsconnect", quietly = TRUE) ||
    packageVersion("rsconnect") < "1.11.0") {
  stop("Install rsconnect >= 1.11.0 before running this script.")
}
accounts <- rsconnect::accounts()
if (!any(accounts$name == "healthify-flu" & accounts$server == "connect.posit.cloud")) {
  stop("First run rsconnect::connectCloudUser() and select healthify-flu.")
}
apps <- read.csv("deployment/apps.csv", stringsAsFactors = FALSE)
stopifnot(nrow(apps) == 5L, !anyDuplicated(apps$content_id))
for (i in seq_len(nrow(apps))) {
  app <- apps[i, ]
  records <- rsconnect::deployments(app$app_path, serverFilter = "connect.posit.cloud")
  if (nrow(records) > 0L) {
    if (nrow(records) != 1L || as.character(records$appId) != app$content_id) {
      stop("Unexpected existing deployment record for ", app$app, "; inspect it before continuing.")
    }
    message(app$app, ": already targets the verified migrated content.")
    next
  }
  rsconnect::migrateToConnectCloud(
    appPath = app$app_path,
    contentId = app$content_id,
    cloudAccount = "healthify-flu",
    appName = app$app
  )
}
