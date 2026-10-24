# Author: Healthify
# Date: 2025-03-11
# Description: A Shiny app to analyze flu vaccine trends and extract flu symptoms from medical notes using NLP
# -------------------------------------------------------------------------------------------------------------
# Libraries
library(shiny)
library(tidyverse)
library(lubridate)
library(tidytext)
library(quanteda)
library(ggwordcloud)
library(viridis)
# -------------------------------------------------------------------------------------------------------------
# Predefined flu symptom keywords
flu_symptoms <- c("fever", "cough", "sore throat", "runny nose", "muscle aches",
                  "fatigue", "headache", "chills", "congestion", "sneezing",
                  "shortness of breath", "body aches", "weakness", "nausea")
# -------------------------------------------------------------------------------------------------------------
# Define UI
ui <- fluidPage(
  titlePanel("Flu Vaccine Trends & Symptom Analysis"),

  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload .txt File with Medical Notes", accept = c(".txt")),
      tags$hr(),
      h4("Instructions:"),
      p("Upload a .txt file containing patient medical notes."),
      p("The app will extract flu symptoms using NLP and display the trends."),
      p("It will also show flu vaccine administration over time."),
      tags$em("Example data: medical_notes_flu_symptom_nlp.txt")  # Italicized text (guide user up demo data)
    ),

    mainPanel(
      tabsetPanel(
        tabPanel("Vaccine Time Series", plotOutput("vaccine_plot")),  # Table removed - not needed/excess content
        tabPanel("Flu Symptoms Analysis",
                 plotOutput("symptom_bar_chart"),
                 plotOutput("word_cloud"))
      )
    )
  )
)
# -------------------------------------------------------------------------------------------------------------
# Define Server Logic
server <- function(input, output, session) {

  # Function to process vaccine records
  process_vaccine_data <- reactive({
    req(input$file)

    text_data <- readLines(input$file$datapath, encoding = "UTF-8")

    data <- tibble(text = text_data) %>%
      mutate(
        age = str_extract(text, "(?<=Age: )\\d+"),
        date = str_extract(text, "(?<=Date: )\\d{4}-\\d{2}-\\d{2}")
      ) %>%
      drop_na() %>%
      mutate(
        age = as.numeric(age),
        date = as.Date(date, format = "%Y-%m-%d")  # Ensures proper Date format
      ) %>%
      filter(!is.na(date) & !is.na(age))

    # Categorize Age Groups (breakdown by large age bands for easy comparison)
    data <- data %>%
      mutate(
        age_group = case_when(
          age < 18 ~ "Children (0-17)",
          age >= 18 & age < 50 ~ "Adults (18-49)",
          age >= 50 & age < 65 ~ "Middle-aged (50-64)",
          age >= 65 ~ "Seniors (65+)"
        )
      )

    return(data)
  })
  # -------------------------------------------------------------------------------------------------------------
  # Function to process flu symptoms using NLP
  extract_flu_symptoms <- reactive({
    req(input$file)

    text_data <- readLines(input$file$datapath, encoding = "UTF-8")

    # Tokenize words from text data
    symptom_words <- tibble(text = text_data) %>%
      unnest_tokens(word, text) %>%
      filter(word %in% flu_symptoms) %>%
      count(word, sort = TRUE)

    return(symptom_words)
  })

  # Generate Vaccine Time Series Plot
  output$vaccine_plot <- renderPlot({
    data <- process_vaccine_data()

    ggplot(data, aes(x = date, fill = age_group)) +
      geom_histogram(binwidth = 7, color = "black", alpha = 0.8, position = "stack") +
      scale_x_date(date_labels = "%b %Y", date_breaks = "1 month") +
      scale_fill_viridis_d(option = "viridis", name = "Age Group") +
      labs(
        title = "Flu Vaccine Doses Administered Over Time",
        x = "Date",
        y = "Number of Vaccines Administered"
      ) +
      theme_minimal()
  })

  # Generate Bar Chart of Symptoms
  output$symptom_bar_chart <- renderPlot({
    symptom_data <- extract_flu_symptoms()

    ggplot(symptom_data, aes(x = reorder(word, -n), y = n, fill = word)) +
      geom_bar(stat = "identity", alpha = 0.9) +
      scale_fill_viridis_d(option = "viridis") +
      labs(title = "Most Common Flu Symptoms in Medical Notes", x = "Symptom", y = "Frequency") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  })

  # Generate Word Cloud of Symptoms
  output$word_cloud <- renderPlot({
    symptom_data <- extract_flu_symptoms()

    ggplot(symptom_data, aes(label = word, size = n, color = n)) +
      geom_text_wordcloud() +
      scale_size_area(max_size = 15) +
      scale_color_viridis_c(option = "viridis") +
      theme_minimal() +
      labs(title = "Word Cloud of Flu Symptoms")
  })
}
# -------------------------------------------------------------------------------------------------------------
# Run the App
shinyApp(ui = ui, server = server)
# -------------------------------------------------------------------------------------------------------------
