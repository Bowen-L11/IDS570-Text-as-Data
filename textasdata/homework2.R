library(readr)
library(dplyr)
library(stringr)
library(tibble)
library(tidytext)

text_dir <- "C:/Users/zhilizhiwai/Documents/Codex/2026-09-27/z-g/outputs/texts"
text_a <- read_file(file.path(text_dir, "A07594__Circle_of_Commerce.txt"))
text_b <- read_file(file.path(text_dir, "B14801__Free_Trade.txt"))
text_c <- read_file(file.path(text_dir, "A06785.txt"))
texts <- tibble(
  doc_title = c("Circle_of_Commerce", "Free_Trade", "Third_Text_A06785"),
  text = c(text_a, text_b, text_c)
)
view(texts)
#step1
texts_clean <- texts %>%
  mutate(
    text_norm = text %>%
      str_replace_all("ſ", "s") %>%
      str_replace_all("\\s+", " ") %>%
      str_to_lower())
words <- texts_clean %>%
  select(doc_title, text_norm) %>%
  unnest_tokens(word, text_norm, token = "words")
data("stop_words")
clean_words <- words %>%
  anti_join(stop_words, by = "word")
View(clean_words)
#step2
bing <- get_sentiments("bing")
sentiment_words <- clean_words %>%
  filter(doc_title %in% c("Circle_of_Commerce", "Free_Trade")) %>%
  inner_join(bing, by = "word")
View(sentiment_words)
#step3
library(tidyr)
raw_sentiment <- sentiment_words %>%
  count(doc_title, sentiment, name = "word_count") %>%
  complete(
    doc_title = c("Circle_of_Commerce", "Free_Trade"),
    sentiment = c("positive", "negative"),
    fill = list(word_count = 0)
  ) %>%
  pivot_wider(
    names_from = sentiment,
    values_from = word_count,
    values_fill = 0
  ) %>%
  mutate(net_sentiment = positive - negative)
#step4
library(quanteda)
token_list <- split(
  clean_words$word,
  clean_words$doc_title)
clean_tokens <- as.tokens(token_list)
dfm_counts <- dfm(clean_tokens)
tfidf_dfm <- dfm_tfidf(dfm_counts)
#step5
tfidf_table <- quanteda::convert(tfidf_dfm, to = "data.frame")
tfidf_words <- tfidf_table %>%
  rename(doc_title = doc_id) %>%
  pivot_longer(
    cols = -doc_title,
    names_to = "word",
    values_to = "tfidf"
  ) %>%
  filter(
    tfidf > 0,
    doc_title %in% c("Circle_of_Commerce", "Free_Trade")  )
#step6
tfidf_sentiment_words <- tfidf_words %>%
  inner_join(bing, by = "word")
top_sentiment_words <- tfidf_sentiment_words %>%
  group_by(doc_title) %>%
  slice_max(order_by = tfidf, n = 10, with_ties = FALSE) %>%
  arrange(doc_title, desc(tfidf)) %>%
  ungroup()
#step7
tfidf_sentiment <- tfidf_sentiment_words %>%
  group_by(doc_title, sentiment) %>%
  summarise(
    total_weight = sum(tfidf),
    .groups = "drop"
  ) %>%
  complete(
    doc_title = c("Circle_of_Commerce", "Free_Trade"),
    sentiment = c("positive", "negative"),
    fill = list(total_weight = 0) ) %>%
  pivot_wider(
    names_from = sentiment,
    values_from = total_weight,
    names_prefix = "tfidf_",
    values_fill = 0) %>%
  mutate(
    tfidf_net = tfidf_positive - tfidf_negative )
#partIII
sentiment_comparison <- raw_sentiment %>%
  rename(
    raw_positive = positive,
    raw_negative = negative,
    raw_net = net_sentiment
  ) %>%
  left_join(tfidf_sentiment, by = "doc_title") %>%
  select(
    doc_title,
    raw_positive, raw_negative, raw_net,
    tfidf_positive, tfidf_negative, tfidf_net)

write_csv(
  sentiment_comparison,
  "homework2_sentiment_comparison.csv")
normalizePath("homework2_sentiment_comparison.csv")




#Q2
word_doc_frequency <- clean_words %>%
  distinct(doc_title, word) %>%
  count(word, name = "document_frequency")

zero_weight_words <- sentiment_words %>%
  count(doc_title, word, sentiment, name = "raw_count") %>%
  left_join(word_doc_frequency, by = "word") %>%
  filter(document_frequency == 3) %>%
  group_by(doc_title) %>%
  slice_max(raw_count, n = 5, with_ties = FALSE) %>%
  ungroup()

View(zero_weight_words)



#Q4
clean_words %>%
  filter(doc_title %in% c("Circle_of_Commerce", "Free_Trade")) %>%
  count(doc_title, name = "cleaned_token_count")
