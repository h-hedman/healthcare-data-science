#title: "infectious_disease_case_summary"
#author: Hayden Hedman
#date: "2021-10-21" ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Load packages -------------------------------------------------------------------------------------------------------------------------------------------------------------------
install.packages(setdiff("pacman", rownames(installed.packages()))) 
## PACKAGE PACMAN LOADS AND CHECKS IF EACH PACKAGE NEEDS INSTALLED
pacman::p_load(dplyr,tidyr, tidyverse, zoo, scales, ggpubr, xlsx)

# Create Dummy Data (remove seed to randomize results) ----------------------------------------------------------------------------------------------------------------------------
set.seed(1)
# Assign variables ----------------------------------------------------------------------------------------------------------------------------------------------------------------
start_date = Sys.Date() - 60
end_date = Sys.Date()
pathogen_list <- c("Influenza", "Measles", "TB", "Pertusis")
location_population = 250000
patient_count = 10000
pop_factor <- function(z){
      ifelse(location_population > 100000, 100000/location_population, location_population/100000)} 

# Create simulated per pathogen ---------------------------------------------------------------------------------------------------------------------------------------------------
flu_data <- data.frame(patient_id  = c(1:patient_count*2),
                            location    = c("Pallet Town"),
                            pathogen    = "Influenza",
                            test_date   = sample(seq(start_date, end_date, by = "day"), patient_count, replace=T),
                            test_result = rbinom(n=patient_count, size=1, prob=0.90))

measles_data <- data.frame(patient_id  = c(1:patient_count*5),
                       location    = c("Pallet Town"),
                       pathogen    = "Measles",
                       test_date   = sample(seq(start_date, end_date, by = "day"), patient_count, replace=T),
                       test_result = rbinom(n=patient_count, size=1, prob=0.60))

tb_data <- data.frame(patient_id  = c(1:patient_count*0.3),
                           location    = c("Pallet Town"),
                           pathogen    = "TB",
                           test_date   = sample(seq(start_date, end_date, by = "day"), patient_count, replace=T),
                           test_result = rbinom(n=patient_count, size=1, prob=0.3))

pertusis_data <- data.frame(patient_id  = c(1:patient_count*0.5),
                      location    = c("Pallet Town"),
                      pathogen    = "TB",
                      test_date   = sample(seq(start_date, end_date, by = "day"), patient_count, replace=T),
                      test_result = rbinom(n=patient_count, size=1, prob=0.5))

comp_pathogen_data <- rbind(flu_data,measles_data,tb_data,pertusis_data)
# Summarize positive pathogen test count by day -----------------------------------------------------------------------------------------------------------------------------------
daily_cases <- data.frame(group_by(comp_pathogen_data, test_date, pathogen) %>% 
                           summarize(count = sum(test_result=="1")))

# Rolling cumulative 7- and 28-day sums -------------------------------------------------------------------------------------------------------------------------------------------
cumulative_data <- with(daily_cases,
                by(daily_cases, pathogen,
                   cumulative_mean <- function(x) {
                     output_data <- x %>%
                       arrange(test_date) %>%
                       mutate(rolling_7d_sum =zoo::rollapplyr(count, 7, sum, partial = TRUE)) 
                     
                     output_data$scaled_incid_100k = signif(output_data$rolling_7d_sum*pop_factor(location_population),digits=2)
                     print(output_data)
                   }))
# Convert dataset to 'data.frame' class -------------------------------------------------------------------------------------------------------------------------------------------
cumulative_df <- do.call(rbind.data.frame, cumulative_data)

# Filter time periods of interest -------------------------------------------------------------------------------------------------------------------------------------------------
sub_rec <- subset(cumulative_df, test_date >= max(test_date-7))
sub_old <- subset(cumulative_df, test_date >= max(test_date-35))

# Summarize mean incidence by last 7 days -----------------------------------------------------------------------------------------------------------------------------------------
sum_rec <- data.frame(group_by(sub_7d, pathogen) %>%
                       summarize(mean_rec_incid = format(mean(scaled_incid_100k), digits=2, nsmall=1)))

# Summarize mean incidence by last 35 days -----------------------------------------------------------------------------------------------------------------------------------------
sum_old <- data.frame(group_by(sub_old, pathogen) %>%
                       summarize(mean_old_incid = format(mean(scaled_incid_100k), digits=2, nsmall=1)))

# Compile table output data --------------------------------------------------------------------------------------------------------------------------------------------------------
table_data <- data.frame(c(sum_rec[1], sum_old[2], sum_rec[2]))
table_data$mean_old_incid <- as.numeric(table_data$mean_old_incid)
table_data$mean_rec_incid <- as.numeric(table_data$mean_rec_incid)
table_data$prop_change <- (table_data$mean_old_incid-table_data$mean_rec_incid)/table_data$mean_old_incid*100
table_data$prop_change <- as.numeric(format(table_data$prop_change, digits=2, nsmall=1))
table_data$Status = "No Change"
table_data$Status[which(table_data$prop_change > 0)] <- "Increasing"
table_data$Status[which(table_data$prop_change < 0)] <- "Decreasing"
clean_table_names <- c("Pathogen", "30-day average incidence per 100,000", "Current 7-day average incidence", "Change (%)", "Status")
names(table_data) <- clean_table_names
# Create output table --------------------------------------------------------------------------------------------------------------------------------------------------------------
summary_table <- ggtexttable(table_data, rows = NULL, 
                             theme = ttheme("light", base_size=12))
# Table caption --------------------------------------------------------------------------------------------------------------------------------------------------------------------
text <- paste("Most recent 7-day period average of incidence compared to previous 30-day period average.",
              "Note: Simulated dummy data not based on literature estimates of example pathogens.", 
              sep = " ")

text_caption <- ggparagraph(text = text, size = 8, color = "black")+
  theme(plot.margin = unit(c(t = -1, r = 1, b = 1, l = 1),"lines"))

# Plot epi curves ------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Rename "Pathogen" for plot legend
colnames(cumulative_df)[2]<-"Pathogen"
# Automated plot title
auto_epi_title <- paste0("Infectious disease Surveillance of", " ", comp_pathogen_data[1,2], ":", " ",  format(start_date, "%m/%d/%y")," ", "-", " ", format(end_date, "%m/%d/%y"))

epi_curve_plot <- ggplot(cumulative_df, aes(x=test_date, y=scaled_incid_100k)) + 
  geom_line(size=3, aes(color=Pathogen))+
  ylab("7-day rolling sum of cases per 100,000") + 
  scale_color_manual(values = c("#E69F00", "#56B4E9", "#332288"))+
  ggtitle(auto_epi_title)+
  theme(text = element_text(size = 14),
        plot.title = element_text(size = 12, face = "bold"),
        panel.background = element_blank(),
        panel.border = element_rect(colour = "black", fill=NA, linewidth=1),
        axis.title.x = element_blank(),
        axis.line = element_line(linewidth = 1, colour = "black"))
# Compile table, text, and plot -----------------------------------------------------------------------------------------------------------------------------------------------------
epi_report_final <- ggarrange(summary_table,text_caption,epi_curve_plot,
                       ncol = 1, nrow = 3,
                       heights = c(1.8,.44,6))
# Automate report file name
fn_report<- paste("Surveillance_Report",comp_pathogen_data[1,2],format(end_date, "%Y-%m-%d"), ".pdf", sep = "")
fn_report <- gsub(" ", "_", fn_report)


# Save report as a PDF for offline dissemination 
ggexport(epi_report_final,filename = fn_report)
# End of script ----------------------------------------------------------------------------------------------------------------------------------------------------------------------






