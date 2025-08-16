# Chicken Movement GIS Data Example
# -------------------------------------------------------------------------------------------------------------------
# Load libraries
library(ggplot2)
library(RColorBrewer)
library(ggmap)
library(Cairo)
library(googlesheets4)
library(maps)
library(raster)
library(sp)
library(rgeos)
# -------------------------------------------------------------------------------------------------------------------
# 1. Load Data 
# -------------------------------------------------------------------------------------------------------------------
# Read chicken movement data from Google Sheets
chicken <- data.frame(gsheet2tbl(
  'https://docs.google.com/spreadsheets/d/1o67qFCOh7dPTsS1iFXG1P_y3eLz_opFYjohoQwIpkLw/edit?usp=sharing'
))
# -------------------------------------------------------------------------------------------------------------------
#  2. Convert UTM to Latitude/Longitude 
# -------------------------------------------------------------------------------------------------------------------
# Remove NA values and select UTM coordinates
chicken_sub <- na.omit(chicken[, c("Easting", "Northing")])

# Convert to spatial points
chicken_sp <- SpatialPoints(chicken_sub)
proj4string(chicken_sp) <- CRS("+proj=longlat")

# Define target projection (UTM Zone 17N, WGS84, km units)
utm_proj <- "+proj=utm +zone=17 +ellps=WGS84 +datum=WGS84 +units=km"

# Transform coordinates
chicken_sp <- spTransform(chicken_sp, CRS(utm_proj))
chicken_convert <- data.frame(chicken_sp)

# Rename columns to Longitude / Latitude
colnames(chicken_convert) <- c("Longitude", "Latitude")

# Combine with original data
chicken2 <- cbind(chicken, chicken_convert)
# -------------------------------------------------------------------------------------------------------------------
# 3. Plot Chicken Movement Density 
# -------------------------------------------------------------------------------------------------------------------
density_plot <- ggplot(data = chicken2, aes(x = Longitude, y = Latitude)) + 
  stat_density2d(aes(fill = after_stat(level)), alpha = 0.7, geom = "polygon") +
  scale_fill_distiller(palette = "Blues", direction = 1) + 
  theme_bw() +
  theme(
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 1),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(colour = "black"),
    legend.position = "none"
  ) +
  ggtitle("Example Chicken Movement Data")

print(density_plot)
# -------------------------------------------------------------------------------------------------------------------
#  4. References
# -------------------------------------------------------------------------------------------------------------------
# YouTube examples:
# https://www.youtube.com/watch?v=kUFlIwpwV6M
# https://www.youtube.com/watch?v=S1fH8pbhcTk
# -------------------------------------------------------------------------------------------------------------------

