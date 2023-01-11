#Factor recoding
library(tidyverse)

# Set my working directory and read in my data
setwd("C:\\users\\bryan\\Documents\\School\\Fall 2022\\Ling")
data = readxl::read_excel("Data Transformation and Descriptive Stats\\Data_ptk.xlsx")
sheet2 = readxl::read_excel("Data Transformation and Descriptive Stats\\Data_ptk.xlsx", sheet = 2)
socio = readxl::read_excel("Data Transformation and Descriptive Stats\\Data_ptk.xlsx", sheet = 3)

view(sheet2)
# Dependent Variables are VOT, COG, and COG2
view(socio)
view(data)

new_data = data %>% 
  left_join(socio, by = c("FILE" = "file")) %>% 
  separate(FILE, c(NA, "spkr_group", "Person")) %>% 
  view()

# IV1: Effect of language (English, Spanish, Heritage)
## On VOT
new_data %>% 
  select(spkr_group, VOT, LANG) %>% 
  group_by(LANG, spkr_group) %>% 
  summarise(count=n(), max=max(VOT), mean=mean(VOT), median=median(VOT), min=min(VOT), range=max(VOT)-min(VOT), std=sd(VOT)) %>% 
  write.table("Data Transformation and Descriptive Stats\\lang_on_VOT.txt")
  #view()

## On COG
new_data %>% 
  select(spkr_group, LANG, COG) %>% 
  group_by(LANG, spkr_group) %>% 
  summarise(count=n(), max=max(COG), mean=mean(COG), median=median(COG), min=min(COG), range=max(COG)-min(COG), std=sd(COG)) %>% 
  view()

## On COG2
new_data %>% 
  select(spkr_group, LANG, COG2) %>% 
  group_by(LANG, spkr_group) %>% 
  summarise(count=n(), max=max(COG2), mean=mean(COG2), median=median(COG2), min=min(COG2), range=max(COG2)-min(COG2), std=sd(COG2)) %>% 
  view()


# IV2: Effect of Sex
## On VOT
new_data %>% 
  select(sex, VOT) %>% 
  group_by(sex) %>% 
  summarise(count=n(), max=max(VOT), mean=mean(VOT), median=median(VOT), min=min(VOT), range=max(VOT)-min(VOT), std=sd(VOT)) %>% 
  view()

## On COG
new_data %>% 
  select(sex, COG) %>% 
  group_by(sex) %>% 
  summarise(count=n(), max=max(COG), mean=mean(COG), median=median(COG), min=min(COG), range=max(COG)-min(COG), std=sd(COG)) %>% 
  view()

## On COG2
new_data %>% 
  select(sex, COG2) %>% 
  group_by(sex) %>% 
  summarise(count=n(), max=max(COG2), mean=mean(COG2), median=median(COG2), min=min(COG2), range=max(COG2)-min(COG2), std=sd(COG2)) %>% 
  view()

# IV3: Effect of AGE
## On VOT
new_data %>% 
  select(age_yr, VOT) %>% 
  group_by(age_yr) %>%  # c("19","21","22","23","26") c("32","33","36","40")
  mutate(age_yr = fct_collapse(as.character(age_yr), less_than_30 = "19", less_than_30 = "21", less_than_30 = "22", less_than_30 = "23", less_than_30 = "26",
                               older_than_30 = "32", older_than_30 = "33", older_than_30 = "36", older_than_30 = "40")) %>% 
  summarise(count=n(), max=max(VOT), mean=mean(VOT), median=median(VOT), min=min(VOT), range=max(VOT)-min(VOT), std=sd(VOT)) %>% 
  view()

## On COG
new_data %>% 
  select(age_yr, COG) %>% 
  group_by(age_yr) %>% 
  mutate(age_yr = fct_collapse(as.character(age_yr), less_than_30 = "19", less_than_30 = "21", less_than_30 = "22", less_than_30 = "23", less_than_30 = "26",
                               older_than_30 = "32", older_than_30 = "33", older_than_30 = "36", older_than_30 = "40")) %>% 
  summarise(count=n(), max=round(max(COG),4), mean=round(mean(COG),4), median=round(median(COG),4), min=round(min(COG),4), range=round(max(COG)-min(COG),4), std=round(sd(COG),4)) %>% 
  write.table("Data Transformation and Descriptive Stats\\age_on_COG.txt")
  #view()

## On COG2
new_data %>% 
  select(age_yr, COG2) %>% 
  group_by(age_yr) %>% 
  mutate(age_yr = fct_collapse(as.character(age_yr), less_than_30 = "19", less_than_30 = "21", less_than_30 = "22", less_than_30 = "23", less_than_30 = "26",
                               older_than_30 = "32", older_than_30 = "33", older_than_30 = "36", older_than_30 = "40")) %>% 
  summarise(count=n(), max=max(COG2), mean=mean(COG2), median=median(COG2), min=min(COG2), range=max(COG2)-min(COG2), std=sd(COG2)) %>% 
  view()

# IV4: Combined Effect of Language and Genre
## On VOT
new_data %>% 
  select(spkr_group,GENRE,VOT) %>% 
  group_by(spkr_group,GENRE,) %>% 
  summarise(count=n(), max=max(VOT), mean=mean(VOT), median=median(VOT), min=min(VOT), range=max(VOT)-min(VOT), std=sd(VOT)) %>% 
  view()

## On COG
new_data %>% 
  select(spkr_group, GENRE, COG) %>% 
  group_by(spkr_group, GENRE) %>% 
  summarise(count=n(), max=max(COG), mean=mean(COG), median=median(COG), min=min(COG), range=max(COG)-min(COG), std=sd(COG)) %>% 
  view()

## On COG2
new_data %>% 
  select(spkr_group, GENRE, COG2) %>% 
  group_by(spkr_group, GENRE) %>% 
  summarise(count=n(), max=max(COG2), mean=mean(COG2), median=median(COG2), min=min(COG2), range=max(COG2)-min(COG2), std=sd(COG2)) %>% 
  view()

# IV5: Combined Effect of Language and Phoneme
## On VOT
new_data %>% 
  select(spkr_group,PHON,VOT) %>% 
  group_by(spkr_group,PHON) %>% 
  summarise(count=n(), max=round(max(VOT),4), mean=round(mean(VOT),4), median=round(median(VOT),4), min=round(min(VOT),4), range=round(max(VOT)-min(VOT),4), std=round(sd(VOT),4)) %>% 
  write.table("Data Transformation and Descriptive Stats\\phon_on_VOT.txt")
  #view()

## On COG
new_data %>% 
  select(spkr_group,PHON,COG) %>% 
  group_by(spkr_group,PHON) %>% 
  summarise(count=n(), max=round(max(COG),4), mean=round(mean(COG),4), median=round(median(COG),4), min=round(min(COG),4), range=round(max(COG)-min(COG),4), std=round(sd(COG),4)) %>% 
  view()

## On COG2
new_data %>% 
  select(LANG, spkr_group,PHON,COG2) %>% 
  summarise(count=n(), max=round(max(COG2),4), mean=round(mean(COG2),4), median=round(median(COG2),4), min=round(min(COG2),4), range=round(max(COG2)-min(COG2),4), std=round(sd(COG2),4)) %>% 
  view()

