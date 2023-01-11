library(tidyverse)

remotes::install_github("joeystanley/joeysvowels")
library(joeysvowels)

midpoints = coronals %>% 
  filter(percent == 50) %>% 
  select(-percent) %>% 
  filter(!vowel %in% c("PRICE", "MOUTH", "CHOICE", "NURSE")) %>% 
  
  print()


# These two do the same thing
ggplot(midpoints, aes(vowel)) + 
  geom_bar()

count(midpoints, vowel) %>% 
  ggplot(aes(vowel,n)) + geom_col()

# Make a plot showing all the syllable onsets
count(midpoints, pre) %>% 
  ggplot(aes(pre,n)) + geom_col()

ggplot(midpoints, aes(pre)) +
  geom_bar()

# Plot for off sets
ggplot(midpoints, aes(fol)) +
  geom_bar()


# Scatterplots
ggplot(midpoints, aes(F2, F1, color=vowel)) +
  geom_point(size = 2, color="black") +
  geom_point(size=1) +
  scale_x_reverse() + 
  scale_y_reverse()

ggplot(midpoints, aes(F2, F1, color=vowel, alpha = end-start, size=2)) +
  geom_point() +
  scale_x_reverse() + 
  scale_y_reverse()

# Terrible to add tons of different shapes
# Scatterplots
terrible = ggplot(midpoints, aes(F2, F1, color=vowel, shape=vowel)) +
  #geom_area() +
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
  #geom_text(data = midpoint_label, aes(label=vowel), size = 4, color="white")
  theme(plot.title.position = "panel") + # "plot" puts it all the way to the left side
  theme(plot.title = element_text(hjust = 0.5, family = "Avenir", face= "italic", color = "yellow", angle = 5)) + # Centers title
  theme(plot.subtitle = element_text(hjust = 1, family = "Iowan Old Style", face = "bold", color = "green", angle = -7, debug=TRUE),
        axis.title.y = element_text(angle=272, vjust = .2, color="blue"),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(color = "lightblue", linetype = "dashed", size = 5),
        panel.background = element_rect(fill = "pink", color = "red", size = 100),
        axis.title.x = element_text(angle = 358, family="Times", face="bold", color ="orange", vjust=.3))
  

## Trend Lines
plot = ggplot(midpoints, aes(F2, F1, color=vowel)) + 
  geom_point() +
  stat_ellipse(level=.75) +
  scale_x_reverse() + 
  scale_y_reverse() +
  labs(x = "second formant frequency",
       y = "vowel height",
       title="Joey's vowel space",
       subtitle = "Notice FACE is higher than KIT",
       caption = "Data available from Joey")


## Theme
plot + theme_bw()
plot + theme_minimal()
plot + theme_test()
plot + theme_classic()
plot + theme_void()
plot + theme_linedraw(base_size = 20, base_family = "Avenir")
plot + theme_gray()
plot + theme_grey()
plot + theme_update()
plot + theme_dark()

# Ugly
plot + see::theme_abyss(base_size = 18) +
  theme(legend.position = "none") +
  #geom_text(data = midpoint_label, aes(label=vowel), size = 4, color="white")
  theme(plot.title.position = "panel") + # "plot" puts it all the way to the left side
  theme(plot.title = element_text(hjust = 0.5, family = "Avenir", face= "italic", color = "yellow", angle = 5)) + # Centers title
  theme(plot.subtitle = element_text(hjust = 1, family = "Iowan Old Style", face = "bold", color = "green", angle = -7, debug=TRUE),
        axis.title.y = element_text(angle=270, vjust = .2),
        panel.grid.minor = element_blank(),
        panel.grid.major = element_line(color = "lightblue", linetype = "dashed", size = 5),
        panel.background = element_rect(fill = "pink", color = "red", size = 100))


getwd()
setwd("C:/Users/bryan/Documents/School/Fall 2022/Ling")
ggsave("ugly_plot.pdf", height = 6, width = 8, device = cairo_pdf)
ggsave("ugly_plot.png", height = 7, width = 8, dpi = 300)

ggsave("terrible.pdf", height = 6, width = 8, device = cairo_pdf)
ggsave("terrible.png", height = 7, width = 8, dpi = 300)


ugly_plot + scale_color_brewer(palette = "set2")
# ugly_plot + scale_color_distiller(palette = "BuGn")
# ugly_plot + scico::scale_color_scico(palette = "buda")
