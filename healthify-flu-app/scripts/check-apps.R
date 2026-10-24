# Smoke-test recovered entry points and important reactive paths before publishing.
source("scripts/project-library.R")
apps <- read.csv("deployment/apps.csv", stringsAsFactors = FALSE)
for (i in seq_len(nrow(apps))) {
  env <- new.env(parent = globalenv())
  entry <- file.path(apps$app_path[i], "app.R")
  parse(entry)
  obj <- source(entry, local = env, encoding = "UTF-8")$value
  stopifnot(inherits(obj, "shiny.appobj"))
  if (apps$app[i] == "app_sir") {
    shiny::testServer(env$server, {
      session$setInputs(R0 = 2.22, duration = 150)
      d <- sir_data()
      stopifnot(nrow(d) == 151L, all(is.finite(as.matrix(d))),
                max(abs(d$S + d$I + d$R - 1000000)) < 0.01,
                !is.null(output$sirPlot))
    })
  } else if (apps$app[i] == "app_geographic") {
    shiny::testServer(env$server, {
      session$setInputs(file = list(datapath = normalizePath("final_processed_data/geographic_incidence_data.csv")), year = "2009")
      stopifnot(nrow(data()) > 0L, !is.null(output$heatmapPlot))
    })
  } else if (apps$app[i] == "app_nlp_symptoms") {
    shiny::testServer(env$server, {
      session$setInputs(file = list(datapath = normalizePath("final_processed_data/medical_notes_flu_symptom_nlp.txt")))
      stopifnot(nrow(process_vaccine_data()) > 0L, nrow(extract_flu_symptoms()) > 0L,
                !is.null(output$vaccine_plot), !is.null(output$symptom_bar_chart), !is.null(output$word_cloud))
    })
  } else if (apps$app[i] == "app_svi") {
    shiny::testServer(env$server, {
      session$setInputs(svi_score = 0.5, interval = "7")
      stopifnot(nrow(epidemic_data()) > 0L, !is.null(output$epiCurvePlot), nzchar(output$sviImpactText))
    })
  } else if (apps$app[i] == "app_vaccine") {
    shiny::testServer(env$server, {
      session$setInputs(vaccine_coverage = 50, interval = "7", show_hospitalizations = FALSE)
      stopifnot(nrow(epidemic_data()) > 0L, !is.null(output$epiCurvePlot), nzchar(output$vaccineImpactText))
      session$setInputs(show_hospitalizations = TRUE)
      stopifnot(!is.null(output$epiCurvePlot))
    })
  }
  message("PASS: ", apps$app[i])
}
