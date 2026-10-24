# Author: Healthify
# Date: 2025-03-10
# Description: A Shiny app to simulate the impact of Social Vulnerability Index (SVI) on an epidemic curve.
# Documentation: Methods based heavily off results from Adams et al. (2024) JAMA on the impact of SVI on epidemic outcomes
# -----------------------------------------------------------------------------------------------------------
# Load necessary libraries
library(shiny)
library(ggplot2)
library(incidence)
# -----------------------------------------------------------------------------------------------------------
# Define UI
ui <- fluidPage(
  titlePanel("Impact of Social Vulnerability Index (SVI) on Epidemic Curve"),

  sidebarLayout(
    sidebarPanel(
      sliderInput("svi_score", "Social Vulnerability Index (SVI)",
                  min = 0, max = 1, value = 0.5, step = 0.01),
      selectInput("interval", "Time Interval:",
                  choices = c("Daily" = 1, "Weekly" = 7, "Monthly" = 30), selected = 7),
      helpText("Higher SVI indicates greater social vulnerability, leading to increased cases and reduced intervention effectiveness.")
    ),

    mainPanel(
      plotOutput("epiCurvePlot"),
      textOutput("sviImpactText")
    )
  )
)

# Define Server
server <- function(input, output, session) {

  # Function to simulate epidemic data based on SVI
  generate_epidemic_data <- function(svi_score) {
    set.seed(8080)

    base_cases <- 3000  # Base epidemic cases
    base_vaccination <- 0.6  # 60% vaccinated in low SVI
    base_antiviral <- 0.2  # 20% receive early antiviral treatment

    # Adjust case scaling based on SVI (Adams et al. found 25% higher ICU/ventilation risk at high SVI)
    case_multiplier <- 1 + (0.25 * svi_score)  # Cases increase with SVI
    total_cases <- round(base_cases * case_multiplier)

    # Adjust vaccination & antiviral rates based on SVI (Adams et al. found -19.4% lower coverage)
    vaccination_rate <- max(0, base_vaccination - (0.194 * svi_score))
    antiviral_rate <- max(0, base_antiviral - (0.173 * svi_score))

    # Simulate case dates
    days <- sample(1:100, total_cases, replace = TRUE, prob = dnorm(1:100, mean = 40, sd = 15))

    data.frame(Date = as.Date("2024-01-01") + days,
               VaccinationRate = vaccination_rate,
               AntiviralRate = antiviral_rate)
  }

  # Reactive function to update epidemic data
  epidemic_data <- reactive({
    generate_epidemic_data(input$svi_score)
  })

  # Render Epidemic Curve Plot
  output$epiCurvePlot <- renderPlot({
    data <- epidemic_data()

    # Compute incidence using `incidence` package
    epi_curve <- incidence(data$Date, interval = as.numeric(input$interval))

    # Set a fixed Y-axis range for consistency
    fixed_y_limit <- c(0, 3000)  # Adjust this based on expected range of cases

    # Plot epidemic curve with fixed Y-axis
    plot(epi_curve, border = "black") +
      theme_minimal(base_size = 14) +
      labs(title = "Epidemic Curve: New Cases Over Time",
           x = "Days",
           y = "New Cases per Time Period") +
      scale_y_continuous(limits = fixed_y_limit)  # Keep Y-axis fixed
  })

  # Render Text Output on SVI Impact
  output$sviImpactText <- renderText({
    data <- epidemic_data()
    paste0(
      "At an SVI of ", input$svi_score,
      ", estimated vaccination coverage is ", round(data$VaccinationRate[1] * 100, 1),
      "% and early antiviral treatment coverage is ", round(data$AntiviralRate[1] * 100, 1),
      "%. This reflects the declining intervention coverage observed in high-SVI groups (Adams et al., 2024)."
    )
  })
}
# -----------------------------------------------------------------------------------------------------------
# Run the app
shinyApp(ui = ui, server = server)
# -----------------------------------------------------------------------------------------------------------
