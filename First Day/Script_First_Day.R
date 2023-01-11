library(tidyverse)
library(ramify)

hl = read_csv("C:\\Users\\bryan\\Documents\\School\\Fall 2022\\Ling\\results.csv")
#view(hl)

filenames = dir("C:/Users/bryan/Documents/School/Fall 2022/Ling/Saints3/", pattern = "\\.txt$", full.names = T)

my_regex = "\\w+(uu) \\1\\w+\\b"

matches = list()
for (file in filenames) {
  cur_text = read_file(file)
  matches = str_extract_all(cur_text, regex(my_regex, ignore_case=T))
  print(matches)
  print(typeof(matches))
  for (match in matches) {
    match
  }
  matches = str_split(matches, " ")
  print(matches)
  
  #matches = append(matches, str_split(str_extract_all(cur_text, regex(my_regex, ignore_case=T)), " "))
}

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

