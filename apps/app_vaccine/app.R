# Author: Healthify
# Date: 2025-03-18
# Description: A Shiny app to simulate the impact of flu Vaccine Coverage on an epidemic curve,
# following key assumptions from Yechezkel et al. (2021).
# ----------------------------------------------------------------------------------------
# Load Packages
library(shiny)
library(ggplot2)
library(incidence)

# ----------------------------------------------------------------------------------------
# Define UI
ui <- fluidPage(
  titlePanel("Impact of Vaccine Coverage on Epidemic Curve"),

  sidebarLayout(
    sidebarPanel(
      sliderInput("vaccine_coverage", "Vaccine Coverage (%)",
                  min = 0, max = 100, value = 50, step = 1),
      selectInput("interval", "Time Interval:",
                  choices = c("Daily" = 1, "Weekly" = 7, "Monthly" = 30), selected = 7),
      helpText("Higher vaccine coverage reduces transmission and hospitalizations."),
      checkboxInput("show_hospitalizations", "Show Estimated Hospitalizations", FALSE)
    ),

    mainPanel(
      plotOutput("epiCurvePlot"),
      textOutput("vaccineImpactText")
    )
  )
)

# ----------------------------------------------------------------------------------------
# Define Server
server <- function(input, output, session) {

  # Function to simulate epidemic data based on vaccine coverage & stratification
  generate_epidemic_data <- function(vaccine_coverage) {
    set.seed(123)

    total_cases <- 5000  # Baseline total epidemic cases

    # Yechezkel et al. stratified by age groups (simplified here)
    age_groups <- c("0-4", "5-19", "20-49", "50-64", "65+")
    transmission_weights <- c(1.2, 1.5, 1.0, 0.8, 0.6)  # Young groups contribute more to transmission
    hospitalization_weights <- c(0.1, 0.3, 0.6, 1.2, 2.5)  # Older groups have higher hospitalization risk

    # Adjust case count based on vaccine coverage
    vaccine_effectiveness <- 0.6  # Assume 60% effectiveness at reducing transmission
    transmission_reduction <- vaccine_coverage / 100 * vaccine_effectiveness
    adjusted_cases <- round(total_cases * (1 - transmission_reduction))

    # Prevent errors at 100% vaccine coverage (no cases left)
    if (adjusted_cases <= 0) {
      return(data.frame(Date = as.Date(character(0)), Hospitalizations = 0))  # Return empty dataset
    }

    # Simulate case dates
    days <- sample(1:100, adjusted_cases, replace = TRUE, prob = dnorm(1:100, mean = 40, sd = 15))

    # Estimate hospitalizations based on Yechezkel's age-stratified assumptions
    age_distribution <- sample(age_groups, adjusted_cases, replace = TRUE, prob = transmission_weights)
    hospitalization_rate <- hospitalization_weights[match(age_distribution, age_groups)] * 0.05  # Adjust base risk

    hospitalizations <- rbinom(adjusted_cases, 1, hospitalization_rate)  # Simulate who gets hospitalized

    data.frame(Date = as.Date("2024-01-01") + days, Hospitalizations = hospitalizations)
  }

  # Reactive function to update epidemic data
  epidemic_data <- reactive({
    generate_epidemic_data(input$vaccine_coverage)
  })

  # Render Epidemic Curve Plot
  output$epiCurvePlot <- renderPlot({
    data <- epidemic_data()

    # Handle case where no infections occur (100% vaccine coverage)
    if (nrow(data) == 0) {
      ggplot() +
        geom_blank() +
        ggtitle("No Epidemic Occurs at 100% Vaccine Coverage") +
        theme_minimal(base_size = 14) +
        labs(x = "Days", y = "New Cases") +
        scale_y_continuous(limits = c(0, 500))  # Safe default Y-axis limit
    } else {
      # Compute incidence using `incidence` package
      epi_curve <- incidence(data$Date, interval = as.numeric(input$interval))

      # Get max cases for dynamic scaling
      max_cases <- max(epi_curve$counts, na.rm = TRUE)

      # Ensure a reasonable Y-axis range (default to 500 if no cases)
      y_limit <- c(0, max(500, max_cases + 50))  # Buffer of 50 for visibility

      p <- plot(epi_curve, border = "black") +
        theme_minimal(base_size = 14) +
        labs(x = "Days", y = "New Cases per Time Period") +
        scale_y_continuous(limits = y_limit)  # Dynamically set Y-axis

      # Overlay estimated hospitalizations if checkbox is checked
      if (input$show_hospitalizations) {
        hospitalization_curve <- incidence(data$Date[data$Hospitalizations == 1], interval = as.numeric(input$interval))
        hospitalization_data <- data.frame(
          Date = hospitalization_curve$dates,
          Cases = rowSums(hospitalization_curve$counts)
        )
        p <- p + geom_line(data = hospitalization_data,
                          aes(x = Date, y = Cases), inherit.aes = FALSE,
                          color = "red", linewidth = 1.2, linetype = "dashed") +
          labs(subtitle = "Red dashed line: Estimated Hospitalizations")
      }

      print(p)
    }
  })

  # Render Text Output on Vaccine Impact
  output$vaccineImpactText <- renderText({
    data <- epidemic_data()
    total_cases <- nrow(data)
    total_hospitalizations <- sum(data$Hospitalizations)

    paste0(
      "At ", input$vaccine_coverage, "% vaccine coverage, estimated total cases are ", total_cases,
      ". Estimated hospitalizations: ", total_hospitalizations,
      ". This model follows the approach of Yechezkel et al. (2021), incorporating age-based transmission and hospitalization risks."
    )
  })
}

# ----------------------------------------------------------------------------------------
# Run the app
shinyApp(ui = ui, server = server)
# ----------------------------------------------------------------------------------------
