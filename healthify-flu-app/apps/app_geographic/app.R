# Author: Healthify
# Date: 2025-03-12
# Goal: Interactive heatmap of median cumulative flu incidence by state
# ---------------------------------------------------------------------------------
# libraries
library(shiny)
library(ggplot2)
library(plotly)
library(dplyr)
library(maps)
library(viridis)
library(readr)
# ---------------------------------------------------------------------------------
# Define UI
ui <- fluidPage(
  titlePanel("US State Heatmap of Flu Incidence"),

  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload CSV File",
                accept = c("text/csv", "text/comma-separated-values,text/plain", ".csv")),
      selectInput("year", "Select Year:", choices = NULL),
      helpText("Upload a CSV file with columns: Year, State (Abbreviation), Median Incidence.")
    ),

    mainPanel(
      plotlyOutput("heatmapPlot"),
      p("Note: States with no available data are shown in dark blue")
    )
  )
)

# Define Server
server <- function(input, output, session) {

  # Load US state map data
  map_data <- map_data("state") %>%
    mutate(region = tolower(region))

  # Reactive function to read uploaded data
  data <- reactive({
    req(input$file)
    df <- read_csv(input$file$datapath)

    # Ensure required columns exist
    required_cols <- c("Year", "State", "Median Incidence")
    if (!all(required_cols %in% colnames(df))) {
      stop("CSV must contain: Year, State, Median Incidence.")
    }

    # Convert state abbreviations to full names for mapping
    state_map <- tibble(
      Abbreviation = state.abb,
      FullName = tolower(state.name)
    )

    df <- df %>%
      left_join(state_map, by = c("State" = "Abbreviation")) %>%
      mutate(State = ifelse(is.na(FullName), tolower(State), FullName)) %>%
      select(Year, State, `Median Incidence`)

    return(df)
  })

  # Update year selection dynamically based on uploaded data
  observe({
    df <- data()
    updateSelectInput(session, "year", choices = unique(df$Year))
  })
  # ---------------------------------------------------------------------------------
  # Render US Heatmap
  output$heatmapPlot <- renderPlotly({
    df <- data()
    req(input$year)

    # Filter data for selected year
    df_filtered <- df %>% filter(Year == input$year)

    # Merge with US state map data to ensure all states are present
    plot_data <- map_data %>%
      left_join(df_filtered, by = c("region" = "State")) %>%
      mutate(`Median Incidence` = ifelse(is.na(`Median Incidence`), 0, `Median Incidence`))

    # Create the heatmap
    p <- ggplot(plot_data, aes(long, lat, group = group, fill = `Median Incidence`,
                               text = paste("State:", region, "<br>Incidence:", `Median Incidence`))) +
      geom_polygon(color = "white") +
      scale_fill_viridis(option = "C", name = "Median Incidence", na.value = "grey30") +
      theme_void() +
      labs(title = paste("Median Confirmed Seasonal Influenza Hospitalization Rate per 100,000 -", input$year))

    ggplotly(p, tooltip = "text")
  })
}
# ---------------------------------------------------------------------------------
# Run the app
shinyApp(ui = ui, server = server)
# ---------------------------------------------------------------------------------
