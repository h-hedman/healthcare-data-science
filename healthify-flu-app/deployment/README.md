# Healthify deployment

The five public app URLs and Connect Cloud IDs are recorded in `apps.csv`.
They were matched to app names in the signed-in Connect Cloud migration dashboard
on October 24, 2026. All five migrations showed Complete.

## Local workspace

The source was recovered from the Healthify Web App folder in Google Drive and
copied into `apps/`. The Drive originals remain as a backup. Each app now has
one explicit `app.R` entry point. Source code, deployment scripts, and public
destination IDs belong in Git. Credentials, local rsconnect records, R history,
and installed packages are ignored.

## Connect this computer once

Open RStudio in this repository and run:

```r
source("scripts/project-library.R")
# rsconnect >= 1.11.0 is required; update it if needed.
rsconnect::connectCloudUser()
# Authorize in your browser as the Healthify account and select healthify-flu.
source("scripts/migrate-deployment-records.R")
```

The migration script writes local records for the existing content IDs. It does
not create new cloud apps or publish code. With the new directory layout, the
records are created at each app directory's root. Copied legacy document records
are retained locally for reference and are not committed.

## Publish an app update

From the repository root, using Rscript on PATH:

```powershell
Rscript --vanilla scripts/check-apps.R
Rscript --vanilla scripts/publish-app.R app_geographic
Rscript --vanilla scripts/publish-app.R app_sir
```

The script accepts one of the five names in `apps.csv` and explicitly targets its
existing ID. Use `app_nlp_symptoms`, `app_svi`, or `app_vaccine` for the others.
After publishing, test the app through its GitHub Pages page, including file uploads
for Geographic and NLP, sliders for SIR/SVI/Vaccine, and the vaccine hospitalization overlay.

## Portfolio updates

Commit and push HTML changes to this repository to update GitHub Pages. Each of
the five iframe sources points directly to its public Connect Cloud URL.
Keep archived shinyapps.io apps intact so previously shared URLs still redirect.

## Recovery findings

- The migrated Geographic and SIR public pages served the Old Faithful starter app
  during the initial check. Recovered source contains the intended heatmap and
  influenza SIR model. Both were republished and their live titles verified.
- The old SIR directory contained a starter `app.R` alongside `app_sir.R`.
  The local canonical entry point uses the actual simulation.
- The recovered vaccine hospitalization overlay failed because its x values did
  not match its counts. The published fix uses incidence dates and explicit mappings.
- All five local smoke checks pass with R 4.5.3. Existing ggplot2/incidence
  deprecation warnings are non-blocking; no modeling redesign was performed.
