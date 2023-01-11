library("rvest")
library("tidyverse")

# assign the output of the pipe to the variable "freqs"
freqs <- "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Hindi_1900" %>% 
  
  # request HTML code
  read_html() %>% 
  
  # find first "table" HTML element
  html_element("table") %>% 
  
  # convert HTML table to data frame
  html_table()

# print first ten rows and number of rows
print(head(freqs, 10))
print(nrow(freqs))
print(tail(freqs))

#//*[@id="mw-content-text"]/div[1]/ol/li[1]
# document.querySelector("#mw-content-text > div.mw-parser-output > ol > li:nth-child(1)")
# <li><span lang="pl"><a href="/wiki/nie#Polish" title="nie">nie</a></span> 6332218</li>

polish_freqs = "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Polish_wordlist" %>% 
  read_html() %>% 
  html_elements(xpath = "//*[@id=\"mw-content-text\"]/div[1]/ol/li") %>% 
  html_text() %>% 
  tibble(both = .) %>% 
  # create a new column "wd" with non-Arabic numeral characters at the beginning of each row in the "both" column
  mutate(wd = str_extract(both, "^\\D+") %>% str_trim()) %>% 
  
  # create a new column "freq" with Arabic numerals
  mutate(freq = str_extract(both, "\\d+") %>% as.numeric()) %>% 
  
  # delete "both" column
  select(-both)
  

print(head(polish_freqs))

# write out results to file on hard drive
write_csv(polish_freqs, file = "C:\\Users\\bryan\\Documents\\School\\Fall 2022\\Ling\\Frequency Lists\\polish_freqs.csv")

danish_freqs <- 
  
  # URL
  "https://en.wiktionary.org/wiki/Wiktionary:Frequency_lists/Danish_wordlist" %>% 
  
  # request the HTML code
  read_html() %>% 
  
  # extract all "li" elements that are children of the "ol" element that is a child of the "div" element that has an attribute of "class" with a value of "mw-parser-output"
  html_elements(xpath = "//div[@class='mw-parser-output']/ol/li") %>% 
  
  # extract the text out of the those "li" elements
  html_text() %>% 
  
  # create a tibble with one column "both"
  tibble(both = .) %>% 
  
  # create a new column "wd" with non-Arabic numeral characters at the beginning of each row in the "both" column
  mutate(wd = str_extract(both, "^\\D+") %>% str_trim()) %>% 
  
  # create a new column "freq" with Arabic numerals
  mutate(freq = str_extract(both, "\\d+") %>% as.numeric()) %>% 
  
  # delete "both" column
  select(-both)

# print first ten lines of frequencies and number of frequencies
print(head(danish_freqs, 10))
print(nrow(danish_freqs))
