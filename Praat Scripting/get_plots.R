library(tidyverse)

# Set my working directory and read in my data
setwd("C:\\users\\bryan\\Documents\\School\\Fall 2022\\Ling\\Praat Scripting")
data = read_csv("intensity_pitch_spreadsheet.csv")

# Remove outliers in Pitch
df = data %>% 
  filter(Pitch<250)

pitch = ggplot(df, aes(x=Pitch)) + 
  geom_histogram() +
  labs(x = "Pitch (Hz)",
       y = "Count",
       title="Pitch of Stressed Vowels",
       caption = "Data from audio reader of 1 Nephi 1")

intensity = ggplot(data, aes(x=Intensity)) + 
  geom_histogram() +
  labs(x = "Intensity (Db)",
       y = "Count",
       title="Intensity of Stressed Vowels",
       caption = "Data from audio reader of 1 Nephi 1")
  

ggsave("pitch.png", plot = pitch, height = 7, width = 8, dpi = 300)
ggsave("intensity.png", plot = intensity, height = 7, width = 8, dpi = 300)
