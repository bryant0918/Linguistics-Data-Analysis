# Load what's needed
library(tidyverse)
remotes::install_github("joeystanley/joeysvowels")
library(joeysvowels)

# Get my data
midpoints = coronals %>% 
  filter(percent == 50) %>% 
  select(-percent) %>% 
  filter(!vowel %in% c("PRICE", "MOUTH", "CHOICE", "NURSE"))

# Terrible plot
terrible = ggplot(midpoints, aes(F2, F1, color=vowel, shape=vowel)) +
  geom_boxplot() +
  geom_rug() +
  scale_color_brewer(palette = "set2") +
  geom_abline() + 
  geom_point(size = 10, color="black") +
  geom_point(size=1) +
  stat_ellipse(level=.75) +
  scale_x_reverse() + 
  scale_y_reverse() + 
  labs(x = "second formant frequency",
       y = "vowel height",
       title="Joey's vowel space",
       subtitle = "Notice FACE is higher than KIT",
       caption = "Data available from Joey") +
  see::theme_abyss(base_size = 18) +
  theme(legend.position = "none") +
  theme(plot.title.position = "panel") + # "plot" puts it all the way to the left side
  theme(plot.title = element_text(hjust = 0.5, family = "Avenir", face= "italic", color = "yellow", angle = 5)) + # Centers title
  theme(plot.subtitle = element_text(hjust = 1, family = "Iowan Old Style", face = "bold", color = "green", angle = -7, debug=TRUE),
        axis.title.y = element_text(angle=272, vjust = .2, color="blue"),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(color = "lightblue", linetype = "dashed", size = 5),
        panel.background = element_rect(fill = "pink", color = "red", size = 100),
        axis.title.x = element_text(angle = 358, family="Times", face="bold", color ="orange", vjust=.3))

# Set my working directory and save my plot
setwd("C:/Users/bryan/Documents/School/Fall 2022/Ling")
ggsave("terrible.pdf", height = 6, width = 8, device = cairo_pdf)
ggsave("terrible.png", height = 7, width = 8, dpi = 300)
