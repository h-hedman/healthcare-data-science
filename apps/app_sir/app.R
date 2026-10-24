# Author: Healthify
# Date: 2025-03-08
# Aim: To simulate an SIR model for seasonal influenza
# -------------------------------------------------------------------------------------------------------------
# Libraries
library(shiny)
library(deSolve)
library(ggplot2)
library(viridis)
library(reshape2)
# -------------------------------------------------------------------------------------------------------------
# Documentation:
# Osthus, D., Hickmann, K. S., Caragea, P. C., Higdon, D., & Del Valle, S. Y. (2017).
# Forecasting seasonal influenza with a state-space SIR model. The annals of applied statistics, 11(1), 202.
# -------------------------------------------------------------------------------------------------------------
# Define SIR model equations
sir_equations <- function(time, variables, parameters) {
  with(as.list(c(variables, parameters)), {
    dS <- -beta * I * S / total_population  # Infection rate
    dI <- beta * I * S / total_population - gamma * I  # Infection minus recovery
    dR <- gamma * I  # Recovery rate
    return(list(c(dS, dI, dR)))
  })
}

# Define UI
ui <- fluidPage(
  titlePanel(HTML("SIR Model Simulation: Adjust R<sub>0</sub>")),

  sidebarLayout(
    sidebarPanel(
      sliderInput("R0", "Basic Reproduction Number (R₀)",
                  min = 1, max = 4, value = 2.22, step = 0.1),
      helpText("Adjust R₀ to reflect flu transmission dynamics."),

      sliderInput("duration", "Simulation Duration (Days)",
                  min = 50, max = 200, value = 150, step = 10),
      helpText("Longer durations help analyze the epidemic curve."),

      tags$hr(),
      strong("Initial Population Settings"),
      textOutput("initS"),
      textOutput("initI"),
      textOutput("initR")
    ),

    mainPanel(
      plotOutput("sirPlot"),
      p("Note: Model parameters are based on Bayesian estimation from Osthus et al. (2017).")
    )
  )
)

# Define Server
server <- function(input, output, session) {

  # Fixed population values
  total_population <- 1000000  # More realistic for a city-level flu outbreak
  initial_S <- total_population * 0.9  # 90% susceptible (Osthus et al. 2017)
  initial_I <- total_population * 0.000172  # Low initial infection rate
  initial_R <- total_population - (initial_S + initial_I)  # Remaining as recovered

  # Define recovery rate from Osthus estimates
  gamma_mean <- 1.7017  # Avg. recovery rate (per week) in Osthus et al. 2017
  gamma <- gamma_mean / 7  # Convert to daily recovery rate (Osthus et al. 2017)

  # Reactive function to solve SIR model
  sir_data <- reactive({
    beta <- input$R0 * gamma  # Compute beta based on R0, Osthus et al. 2017
    beta <- rnorm(1, mean = beta, sd = 0.05 * beta)  # Introduce small variability in β, Osthus et al. 2017
    gamma_sample <- rnorm(1, mean = gamma, sd = 0.02 * gamma)  # Introduce variability in γ, Osthus et al. 2017

    # Parameter list
    parameters_values <- c(beta = beta, gamma = gamma_sample, total_population = total_population)
    initial_values <- c(S = initial_S, I = initial_I, R = initial_R)
    time_values <- seq(0, input$duration, by = 1)  # User-selected simulation duration

    sir_solution <- ode(y = initial_values, times = time_values,
                        func = sir_equations, parms = parameters_values)
    as.data.frame(sir_solution)
  })

  # Render Initial Values
  output$initS <- renderText({ paste("Susceptible:", formatC(initial_S, format = "d", big.mark = ",")) })
  output$initI <- renderText({ paste("Infected:", formatC(initial_I, format = "d", big.mark = ",")) })
  output$initR <- renderText({ paste("Recovered:", formatC(initial_R, format = "d", big.mark = ",")) })

  # Render SIR Plot
  output$sirPlot <- renderPlot({
    df <- sir_data()

    # Convert data to long format for ggplot
    df_long <- melt(df, id = "time")

    ggplot(df_long, aes(x = time, y = value, color = variable)) +
      geom_line(size = 1.5) +
      scale_color_viridis(discrete = TRUE, name = "Population Groups") +
      labs(title = "SIR Model Simulation for Seasonal Influenza",
           x = "Time (days)",
           y = "Population",
           color = "Group") +
      theme_minimal(base_size = 14) +
      theme(legend.position = "top")
  })
}
# -------------------------------------------------------------------------------------------------------------
# Run the app
shinyApp(ui = ui, server = server)
# -------------------------------------------------------------------------------------------------------------
