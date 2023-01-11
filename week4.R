library(tidyverse)

my_data <- tibble(
  person = c("Bob", "Bob", "Billy", "Billy", "Britta", "Britta", "Bonnie"), 
  wd = c("apple", "banana", "orange", "mango", "apple", "manzana", "kiwi")
)

sociodemo <- tibble(
  person = c("Bonnie", "Billy", "Bob"), 
  age = c(47, 6, 95)
)

freqs <- tibble(
  wd = c("apple", "banana", "kiwi", "mango"), 
  freq = c(123, 234, 345, 456)
)

my_data %>% 
  left_join(sociodemo, by = c("person" = "person")) %>% 
  left_join(freqs, by = c("wd" = "wd"))

polish_freqs_file <- dir("C:\\Users\\bryan\\Documents\\School\\Fall 2022\\Ling\\Frequency Lists\\", pattern = ".csv$", full.names = TRUE)
print(polish_freqs_file)

filenames <- dir("C:\\Users\\bryan\\Documents\\School\\Fall 2022\\Ling\\Saints3", pattern = ".txt$", full.names = TRUE)

ly_words_file <- dir("C:\\Users\\bryan\\Documents\\School\\Fall 2022\\Ling\\", pattern = "Saints_results.csv$", full.names = TRUE)

print(ly_words_file)
ly = read_csv(ly_words_file)
view(ly)

all_wds = c()
for (file in filenames) {
  cur_txt = read_file(file) %>% str_to_upper()
  cur_wds = str_extract_all(cur_txt, regex("[a-z-'’]+", ignore_case = T)) %>% unlist()
  all_wds = append(all_wds, cur_wds)
}

freqs = table(all_wds)

freqs_df = tibble(wd = names(freqs), freq = freqs)

print(freqs_df)

freqs_df %>% arrange(desc(freq))
print(freqs_df)

my_dataframe = left_join(ly, freqs)

#danish_data_file = dir("C:\\Users\\bryan\\Downloads\\Dataset_for_over_dan.csv", pattern="", full.names=TRUE)
#freqs_data_file = dir("C:\\Users\\bryan\\Downloads\\freqs_dan.csv", full.names = TRUE)

danish_data = read.csv("C:\\Users\\bryan\\Downloads\\Dataset_for_over_dan.csv") %>% mutate(node = str_to_upper(node))
freqs_data = read.csv("C:\\Users\\bryan\\Downloads\\freqs_dan.csv") %>% mutate(wd = str_to_upper(wd))

my_dataframe = left_join(danish_data, freqs_data, by = c("node" = "wd"))
view(my_dataframe)


library("reticulate")
py_install("num2words")
num2words = import("num2words")
print(num2words$num2words(53))

txt = "Narodziłem się w roku 1988, 18 Września."
str_replace_all(txt, "\\d+", function(x) num2words$num2words(x, lang="pl"))
print(num2words$num2words(txt, lang="pl"))

txt = "I was born in the year 1998 on September 18."
str_replace_all(txt, "\\d+", num2words$num2words)
print(num2words$num2words(txt))

## Lesson 4.3
setwd("C:\\users\\bryan\\Downloads")

span = read_csv("data_coded.csv")
print(count(span,when))

span %>% 
  #filter(when %in% c("begin", "after")) %>% 
  arrange(FILE, -S_START) %>% 
  mutate(upper_word = str_to_upper(WORD), .after = WORD) %>% 
  select(upper_word, everything()) %>% 
  select(FILE, country, when, starts_with("S", ignore.case = F)) %>% 
  #select(matches("^s", ignore.case = F)) %>% 
  group_by(country, when) %>% 
  mutate(when, recode(when, after = "end")) %>% 
  summarise(n(), mean(S_DUR), median(S_DUR))


spain = span %>% filter(country == "Spain")
view(span)
native_spanish = spain %>% filter(when == "native")
view(native_spanish)

native_spanish_file = select(native_spanish, -FILE, -country)
view(native_spanish_file)

## September 23
setwd("C:\\users\\bryan\\Documents\\School\\Fall 2022\\Ling")
span = readxl::read_excel("data_L2s.xlsx", sheet = "data_hls_2017-10-25")
view(span)

span %>% 
  filter(when %in% c("begin", "end")) %>% 
  arrange(FILE, -S_START) %>% 
  mutate(upper_word = str_to_upper(WORD), .after = WORD) %>% 
  select(upper_word, everything()) %>% 
  select(FILE, country, when, starts_with("S", ignore.case = F)) %>% 
  #select(matches("^s", ignore.case = F)) %>% 
  group_by(country, when) %>% 
  mutate(when = recode(when, end = "after")) 

span %>% 
  summarise(n(), mean(S_DUR), median(S_DUR))

span %>% 
  arrange(word) %>% 
  select(word, everything()) %>% 
  view()












