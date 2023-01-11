library(tidyverse)

# Set my working directory and read in my data
setwd("C:\\users\\bryan\\Documents\\School\\Fall 2022\\Ling")
span = readxl::read_excel("data_L2s.xlsx", sheet = "data_hls_2017-10-25")
view(span)

# First transformation
span = span %>% 
  filter(when %in% c("begin", "end")) %>%  # Only grab beginning and ending values
  arrange(FILE, -S_START) %>%             # Arrange by file and negative S_START
  # Create another collumn of uppercase words right after the word column
  mutate(upper_word = str_to_upper(WORD), .after = WORD) %>% 
  # Now grab certain columns including ones that start with the letter S
  select(FILE, country, when, WORD, starts_with("S", ignore.case = F)) %>% 
  group_by(country, when) %>% # Group by country and when
  mutate(when = recode(when, end = "after")) # Change the values "end" to "after"

# Summary from my first transformation
span %>% 
  summarise(n(), mean(S_DUR), median(S_DUR))

view(span)

# Arrange it by word and descending order of duration
span = span %>% 
  arrange(WORD, desc(S_DUR)) %>% 
  select(FILE, WORD, everything()) %>% # Put file and word at the beginning and save
  view()

# Unite s_start and S_end into a single column called "Interval"
span %>%
  unite(col="Interval", S_START, S_END, sep=" : ") %>% 
  view()

# Pivot everything under country and SA country into two new columns "Spain" and "DR"
span = span %>% 
  pivot_wider(names_from = 'country', values_from = `SA country`) %>% 
  view()

# Undo my last pivot but rename the columns still slightly.
span %>% 
  pivot_longer(DR:Spain, names_to="Country", values_to="Study Abroad") %>% 
  view()
