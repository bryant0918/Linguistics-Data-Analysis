# R script to retrieve determiners and following words from Proverbs 1 and 2 in Hiligaynon
# Earl Kjar Brown, ekbrown byu edu (add characters to create email)

# load the workhorse package (actually packages)
library("tidyverse")

# define function to remove newlines and tab breaks from a string
clean_str <- function(input_str) {
  output <- input_str
  output <- str_replace_all(output, "\\n", " ")
  output <- str_replace_all(output, "\\t", " ")
  return(output)
}

# specify the regular expression ("regex" for short)
# up to 20 characters of preceding context, the determiner, the next word, up to 20 characters of following context
my_regex <- "(.{0,20})\\b(s?ang (?!=mga)|sing|sa|s?ang mga|si|ni|kay|sanday|nanday|kanday)\\W+(\\w+)(.{0,20})"

# get filenames
filenames <- dir("/Users/ekb5/Downloads", pattern = "_hlgn.txt$", full.names = TRUE)

# create collector data frame
all_matches_df <- data.frame()

# loop over filenames
for (filename in filenames) {
  
  # print progress report to user
  cat("Working on", filename, "\n")
  
  # slurp all text into a single character string
  cur_txt <- read_file(filename)
  
  # replace newlines and tab breaks (if any) with spaces
  cur_txt <- clean_str(cur_txt)
  
  # extract matches out of text (returns a list)
  cur_matches_list <- str_match_all(cur_txt, regex(my_regex, ignore_case = TRUE))  
  
  # index into the first element of the list, which is a matrix
  cur_matches_matrix <- cur_matches_list[[1]]
  
  # loop over the rows of the matrix
  for (i in 1:nrow(cur_matches_matrix)) {
    
    # create one-row data frame of current match
    cur_match_df<- data.frame(
      file = basename(filename),  # filename
      full_context = cur_matches_matrix[i, 1],  # full context
      pre_context = cur_matches_matrix[i, 2],  # preceding context
      determiner = cur_matches_matrix[i, 3],  # determiner
      next_wd = cur_matches_matrix[i, 4],  # next word
      post_context = cur_matches_matrix[i, 5]  # following context
      )
    
    # bind the current match to the collector data frame 
    all_matches_df <- bind_rows(all_matches_df, cur_match_df)
    
  }  # next match (next iteration in i for loop)
  
}  # next filename (next iteration in filename for loop)

# write out results to file on hard drive
write_csv(all_matches_df, file = "/Users/ekb5/Downloads/results.csv")
