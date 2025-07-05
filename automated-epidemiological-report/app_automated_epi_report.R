# ---------------------------------------------------------------------------
# Author: Hayden Hedman
# Date: 2021-10-16
# Description: A Shiny app to generate automated epidemiological reports for
# infectious diseases with upload, visualizations, and PDF export.
# ---------------------------------------------------------------------------
# Load packages
library(shiny)
library(dplyr)
library(zoo)
library(ggplot2)
library(ggpubr)
library(readr)
library(readxl)
library(tools)
library(viridis)
# ---------------------------------------------------------------------------
# Simulate fallback data
simulate_data <- function(patient_count = 1000, location = "Pallet Town") {
  set.seed(1)
  start_date <- Sys.Date() - 60
  end_date <- Sys.Date()
  pathogens <- c("Influenza", "Measles", "TB", "Pertusis")
  
  data.frame(
    patient_id  = 1:patient_count,
    location    = location,
    pathogen    = sample(pathogens, patient_count, replace = TRUE),
    test_date   = sample(seq(start_date, end_date, by = "day"), patient_count, replace = TRUE),
    test_result = rbinom(n = patient_count, size = 1, prob = 0.5)
  )
}
# ---------------------------------------------------------------------------
# UI
ui <- fluidPage(
  titlePanel("Infectious Disease Case Report Generator"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file_input", "Upload Case Data (CSV/XLSX)", accept = c(".csv", ".xlsx")),
      numericInput("population", "Population Size", value = 250000),
      uiOutput("pathogen_ui"),
      dateRangeInput("date_range", "Select Date Range", start = Sys.Date() - 60, end = Sys.Date()),
      actionButton("generate_report", "📄 Generate PDF Report", class = "btn-primary"),
      br(), br(),
      downloadButton("download_report", "Download Report")
    ),
    mainPanel(
      textOutput("status_msg"),
      tableOutput("concern_table"), 
      plotOutput("preview_plot", height = "400px")
    )
  )
)
# ---------------------------------------------------------------------------
# Server
server <- function(input, output, session) {
  # Read uploaded data
  input_data <- reactive({
    req(input$file_input)
    ext <- tools::file_ext(input$file_input$name)
    if (ext == "csv") {
      read_csv(input$file_input$datapath, show_col_types = FALSE)
    } else if (ext %in% c("xls", "xlsx")) {
      read_excel(input$file_input$datapath)
    } else {
      showNotification("Unsupported file type.", type = "error")
      return(NULL)
    }
  })
  # ---------------------------------------------------------------------------
  # Use uploaded or fallback simulated data
  data <- reactive({
    if (is.null(input$file_input)) {
      simulate_data()
    } else {
      df <- input_data()
      req(all(c("test_date", "test_result", "pathogen") %in% names(df)))
      df$test_date <- as.Date(df$test_date)
      df <- df[!is.na(df$test_date), ]
      df
    }
  })
  # ---------------------------------------------------------------------------
  # Pathogen selector
  output$pathogen_ui <- renderUI({
    req(data())
    selectInput("pathogens", "Select Pathogens",
                choices = unique(data()$pathogen),
                selected = unique(data()$pathogen),
                multiple = TRUE)
  })
  # ---------------------------------------------------------------------------
  # Summary stats table
  summary_stats <- reactive({
    df <- data()
    df <- df[df$pathogen %in% input$pathogens &
               df$test_date >= input$date_range[1] &
               df$test_date <= input$date_range[2], ]
    
    if (nrow(df) == 0 || all(is.na(df$test_date))) return(NULL)
    
    daily <- df %>%
      group_by(test_date, pathogen) %>%
      summarize(count = sum(test_result == 1), .groups = "drop")
    
    roll_df <- daily %>%
      group_by(pathogen) %>%
      arrange(test_date) %>%
      mutate(
        rolling_7d = zoo::rollapplyr(count, 7, sum, fill = 0, partial = TRUE),
        rolling_28d = zoo::rollapplyr(count, 28, sum, fill = 0, partial = TRUE),
        per_100k_7d = round(rolling_7d * (100000 / input$population), 1),
        per_100k_28d = round(rolling_28d * (100000 / input$population), 1)
      ) %>%
      ungroup()
    
    valid_dates <- roll_df$test_date[!is.na(roll_df$test_date)]
    if (length(valid_dates) == 0) return(NULL)
    
    latest_date <- max(valid_dates)
    if (!is.finite(latest_date)) return(NULL)
    
    latest <- roll_df %>%
      filter(test_date == latest_date) %>%
      mutate(
        change = per_100k_7d - per_100k_28d,
        concern = case_when(
          per_100k_7d > 25 ~ "High",
          per_100k_7d > 10 ~ "Medium",
          TRUE ~ "Low"
        )
      ) %>%
      select(Pathogen = pathogen,
             `7-day Incidence` = per_100k_7d,
             `28-day Incidence` = per_100k_28d,
             `Change` = change,
             `Concern` = concern)
    
    latest
  })
  # ---------------------------------------------------------------------------
  # Show table - TEMP
  output$concern_table <- renderTable({
    req(summary_stats())
    summary_stats()
  }, striped = TRUE, digits = 1)
  # ---------------------------------------------------------------------------
  # Show plot
  output$preview_plot <- renderPlot({
    df <- data()
    req(input$pathogens)
    df <- df[df$pathogen %in% input$pathogens &
               df$test_date >= input$date_range[1] &
               df$test_date <= input$date_range[2], ]
    
    if (nrow(df) == 0) return(NULL)
    
    summary <- df %>%
      group_by(test_date, pathogen) %>%
      summarize(count = sum(test_result == 1), .groups = "drop") %>%
      group_by(pathogen) %>%
      arrange(test_date) %>%
      mutate(rolling_7d = zoo::rollapplyr(count, 7, sum, fill = NA, partial = TRUE),
             per_100k = round(rolling_7d * (100000 / input$population), 2))
    
    ggplot(summary, aes(x = test_date, y = per_100k, color = pathogen)) +
      geom_line(size = 2.2, alpha = 0.7) +
      scale_color_viridis_d(option = "D", end = 0.85) +
      labs(title = "Rolling 7-Day Incidence per 100,000",
           y = "Cases per 100,000", x = "Date") +
      theme_minimal(base_size = 14) +
      theme(legend.title = element_blank())
  })
  # ---------------------------------------------------------------------------
  # PDF Export
  report_path <- reactiveVal(NULL)
  
  observeEvent(input$generate_report, {
    req(summary_stats())
    df <- data()
    df <- df[df$pathogen %in% input$pathogens &
               df$test_date >= input$date_range[1] &
               df$test_date <= input$date_range[2], ]
    
    if (nrow(df) == 0) return(NULL)
    
    timeseries <- df %>%
      group_by(test_date, pathogen) %>%
      summarize(count = sum(test_result == 1), .groups = "drop") %>%
      group_by(pathogen) %>%
      arrange(test_date) %>%
      mutate(rolling_7d = zoo::rollapplyr(count, 7, sum, fill = NA, partial = TRUE),
             per_100k = round(rolling_7d * (100000 / input$population), 2))
    
    table_plot <- ggtexttable(summary_stats(), rows = NULL, theme = ttheme("light", base_size = 12))
    line_plot <- ggplot(timeseries, aes(x = test_date, y = per_100k, color = pathogen)) +
      geom_line(size = 2.2, alpha = 0.7) +
      scale_color_viridis_d(option = "D", end = 0.85) +
      labs(title = "Rolling 7-Day Incidence per 100,000",
           y = "Cases per 100,000", x = "Date") +
      theme_minimal(base_size = 14) +
      theme(legend.title = element_blank())
    caption <- ggparagraph("Note: Summary reflects most recent 7-day and 28-day incidence. Simulated or uploaded data.", size = 9)
    
    ##layout <- ggarrange(table_plot, line_plot, caption, ncol = 1, heights = c(1.8, 4, 0.6))
    layout <- ggarrange(line_plot, caption, ncol = 1, heights = c(6, 0.6))
    
    
    filename <- paste0("report_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".pdf")
    ggexport(layout, filename = filename)
    report_path(filename)
    output$status_msg <- renderText("✅ Report successfully generated.")
  })
  # ---------------------------------------------------------------------------
  # Enable download
  output$download_report <- downloadHandler(
    filename = function() basename(report_path()),
    content = function(file) file.copy(report_path(), file)
  )
}
# ---------------------------------------------------------------------------
# Launch app
shinyApp(ui, server)
# ---------------------------------------------------------------------------

