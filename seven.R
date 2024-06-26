---
  title: 'Worksheet 7: Topic Modeling'
author: 'William David Hiott, Sr.'
date: 'April 15, 2024'
---
  
  _This is the seventh in a series of worksheets for History 8510 at Clemson University. The goal of these worksheets is simple: practice, practice, practice. The worksheet introduces concepts and techniques and includes prompts for you to practice in this interactive document. When you are finished, you should change the author name (above), knit your document, and upload it to canvas. Don't forget to commit your changes as you go and push to github when you finish the worksheet._

Text analysis is an umbrella for a number of different methodologies. Generally speaking, it involves taking a set (or corpus) of textual sources, turning them into data that a computer can understand, and then running calculations and algorithms using that data. Typically, at its most basic level, that involves the counting of words.

Topic modeling (TM) is one type of text analysis that is particularly useful for historians. 

TM takes collections or corpuses of documents and returns groups of "topics" from those documents. It is a form of unsupervised classification that finds groups of items that are probabilistically likely to co-occur. 

Latent Dirichlet allocation (LDA) is the most popular algorithm or method for topic modeling, although there are others. It assumes that each document has a mixture of topics and that each topic is a mixture of words. That means that topics overlap each other in terms of content rather than being confined to distinct and singular groups. 

To prepare a corpus for topic modeling, we'll do many of the same types of operations that we used last week to prepare a corpus for analysis. First we'll pre-process the data and then we'll create a document term matrix from our corpus using the `tm` (text mining) package. 

```{r}
library(tidytext)
library(tidyverse)
library(readtext)
library(tm)
library(topicmodels)

```

```{r}
download.file("https://github.com/regan008/8510-TextAnalysisData/blob/main/TheAmericanCity.zip?raw=true", "AmCity.zip")
unzip("AmCity.zip")

# Metadata that includes info about each issue.
metadata <- read.csv("https://raw.githubusercontent.com/regan008/8510-TextAnalysisData/main/AmCityMetadata.csv")

meta <- as.data.frame(metadata)
#meta$Filename <- paste("MB_", meta$Filename, sep="")
file_paths <- system.file("TheAmericanCity/")
ac_texts <- readtext(paste("TheAmericanCity/", "*.txt", sep=""))
ac_whole <- full_join(meta, ac_texts, by = c("filename" = "doc_id")) %>% as_tibble() 

tidy_ac <- ac_whole %>%
  unnest_tokens(word, text) %>% 
  filter(str_detect(word, "[a-z']$")) %>% 
  anti_join(stop_words)
```
The above code borrows from what we did last week. It pulls in the texts from the _The American City_ corpus, joins them together into a single data frame, and then turns then uses `unnest_tokens()` to tokenize the text and, finally, removes stop words. 

For topic modeling, we need a Document Term Matrix, or a DTM. Topic Modeling has the documents running down one side and the terms across the top. `Tidytext` provides a function for converting to and from DTMs. First, we need to create a document that has the doc_id, the word and the count of the number of times that word occurs. We can do that using `count()`.

```{r}
tidy_ac_words <- tidy_ac %>% count(filename, word)
```

Now we can use `cast_dtm()` to turn `tidy_mb_words` into a dtm. 

```{r}
ac.dtm <- tidy_ac_words %>% 
  count(filename, word) %>% 
  cast_dtm(filename, word, n)
```

If you run `class(mb.dtm)` in your console you will notice that it now has a class of "DocumentTermMatrix". 

Now that we have a dtm, we can create a topic model. For this, we'll use the topic models package and the `LDA()` function. Take a minute and read the documentation for `LDA()`.

There are two important options when running `LDA()`. The first is k which is the number of topics you want the model to generate. What number topics you generate is a decision that often takes some experimentation and depends on the size of your corpus. The American City corpus isn't that bigbut still has over 209k words. In this instance, because the corpus is so small we're going to start with a small number of topics. Going above 5 causes errors with this particular corpus. Later, when you work with a different corpus you should experiment with changing the number of topics from 10 to 20 to 30 to 50 to see how it changes your model. 

The second important option when running `LDA()` is the seed option. You don't worry too much about what setting the seed does, but put simply - it ensures the output of the model is predictable and reproducible. Using the seed ensures that if you come back to your code later or someone else tries to run it, the model will return exactly the same results. 

Lets now train our model. This will take a few minutes: 
  ```{r}
ac.lda <- LDA(ac.dtm, k = 5, control = list(seed = 12345))
ac.lda
```

Now we have a LDA topic model that has 5 topics. There are two ways to look at this model: word-topic probabilities and document-topic probabilities. 

Lets start with **word-topic probabilities.**
  
  Every topic is made up of words that are most associated with that topic. Together these words typically form some sort of theme. To understand what this looks like the easiest thing to do is create a bar chart of the top terms in a topic. 

```{r}
ac.topics <- tidy(ac.lda, matrix = "beta")
ac.topics
```
What we have here is a list of topics and the weight of each term in that topic. Essential we have turned this into a one-topic-per-term-per-row format. So, for example, the term 10th has a weight of 5.135047e-05 in topic 1 but 7.269700e-05 in topic 2. Now that doesn't mean a lot to us at this moment and this format is impossible to grasp in its current size and iteration, but we can use tidyverse functions to pair this down and determine the 10 terms that are most common within each topic. 
```{r}
ac.top.terms <- ac.topics %>%
  group_by(topic) %>%
  top_n(5, beta)

ac.top.terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()
```
(@) Can you adjust the code above to show the top 10 words from just one topic?
```{r}
ac.top.terms <- ac.topics %>%
  arrange(desc(beta)) %>% 
  group_by(topic) %>% slice(1:10)

ac.top.terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()
```
Ran ten of all five topics.

Another useful way to look at the words in each topic is by visualizing them as a wordcloud.
```{r warning=FALSE}
library(wordcloud)
topic1 <- ac.topics %>% filter(topic == 2)
wordcloud(topic1$term, topic1$beta, max.words = 100, random.order = FALSE,
    rot.per = 0.3, colors = brewer.pal(6, "Dark2"))
```

Now we can see what words are most common in each topic. But the document-topic probabilities are also useful for understanding what topics are prevalent in what documents. Just as each topic is made up of a mixture of words, the LDA algorithm also assumes that each topic is made up of a mixture of topics. 

```{r}
ac.documents <- tidy(ac.lda, matrix = "gamma")
ac.documents
```
For each document, the model gives us an estimated proportion of what words in the document are from a topic. So for the April 1915 issue it estimates that about 23% of the words are from topic 1. 

This is easier to see if we filter to see the breakdown for just one document. 
```{r}
ac.documents %>%  filter(document == "1916_May.txt") %>% arrange(desc(gamma))
```

(@) Use the rest of this worksheet to experiment with topic modeling. I've added the code to download a much larger dataset - the issues of Mind and Body. This corpus has 413 documents ranging from the 1890s to 1936. You'll want to start with at least 25 topics. 
```{r}
download.file("https://github.com/regan008/8510-TextAnalysisData/blob/main/MindAndBody.zip?raw=true", "MB.zip")
unzip("MB.zip")

# Metadata that includes info about each issue.
mbmetadata <- read.csv("https://raw.githubusercontent.com/regan008/8510-TextAnalysisData/main/mb-metadata.csv")
```
```{r}
download.file("https://github.com/regan008/8510-TextAnalysisData/blob/main/MindAndBody.zip?raw=true", "MB.zip")
unzip("MB.zip")
file.rename("txt", "mbtxt")
```

(@) What happens if you create a custom stopword list? How does this change the model?

This is just one example of stopwords. You can find other lists such as stopwords in other languages or [stopwords designed specifically for the 19th century.](https://www.matthewjockers.net/macroanalysisbook/expanded-stopwords-list/) Its also possible you may want to edit the list of stopwords to include some of your own. For example, if we wanted to add the word, "America" to the stopwords list we could use add_row to do so: 

```{r}
stop_words_custom <- stop_words %>% add_row(word="America", lexicon="NA")
```

(@) Can you create a topic model for just the documents in the 1920s? How does that change the model? 
```{r}
mbmeta <- as.data.frame(mbmetadata)
meta$Filename <- paste("MB_", meta$Filename, sep="")
file_paths <- system.file("MindandBody/")
mbmeta_mbtexts <- readtext(paste("MindandBody/", "*.mbtxt", sep=""))
mbmeta_whole <- full_join(mbmeta, mb_texts, by = c("filename" = "doc_id")) %>% as_tibble() 

tidy_mbmeta <- mbmeta_whole %>%
  unnest_tokens(word, text) %>% 
  filter(str_detect(word, "[a-z']$")) %>% 
  anti_join(stop_words)
```

```{r}
tidy_mbmeta <- mbmeta_whole %>%
  unnest_tokens(word, mbtext) %>% 
  filter(str_detect(word, "[a-z']$")) %>% 
  anti_join(stop_words)
```

```{r}
# Metadata that includes info about each issue.


mbmetadata <- as.data.frame(mbmetadata)
meta$Filename <- paste("MB_", meta$Filename, sep="")
file_paths <- system.file("Mind and Body")
mbmeta_texts <- readtext(paste("Mind and Body", "*.txt", sep=""))
mbmeta_whole <- full_join(mbmetadata, mb_texts, by = c("filename" = "doc_id")) %>% as_tibble()
```


```{r}
tidy_mbmeta <- mbmeta_whole %>%
  unnest_tokens(word, text) %>% 
  filter(str_detect(word, "[a-z']$")) %>% 
  anti_join(stop_words)
```

```{r}
tidy_mbmeta_words <- tidy_mbmeta %>% count(filename, word)
```


```{r}
mbmeta.dtm <- tidy_mbmeta_words %>% 
  count(filename, word) %>% 
  cast_dtm(filename, word, n)
```


```{r}
mbmeta.lda <- LDA(mbmeta.dtm, k = 5, control = list(seed = 12345))
mbmeta.lda
```

```{r}
mbmeta.lda <- LDA(mbmeta.dtm, k = 25, control = list(seed = 12345))
mbmeta.lda
```

```{r}
mbmeta.topics <- tidy(mbmeta.lda, matrix = "beta")
mbmeta.topics
```

```{r}
mb.top.terms <- mb.topics %>%
  group_by(topic) %>%
  top_n(25, beta)

mb.top.terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()
```

```{r warning=FALSE}
library(wordcloud)
topic1 <- mbmetadata.topics %>% filter(topic == 2)
wordcloud(topic1$term, topic1$beta, max.words = 100, random.order = FALSE,
    rot.per = 0.3, colors = brewer.pal(6, "Dark2"))
```


```{r}
mb.documents <- tidy(ac.lda, matrix = "gamma")
mb.documents
```
Above ran.

```{r}
mb.documents %>%  filter(document == "1915_April.txt") %>% arrange(desc(gamma))
```
Above ran.

```{r}

```



(@) Now, lets return to the Buffalo Bill data from last week. You should be able to use topic modeling to address two of the research questions provided:

* Can we detect some change over time in promotion language and reception language (marketing and reviewing)? 

1. QUESTION AS TO MARKETING AND REVIEWING
Were there types of characters, scenarios, action promised in promotional material and/or noted in reviews earlier vs later?
2 CHARACTERS, SCENARIOS, ACTIONS
* What can be gleaned from the items tagged as extraneous as far as topics? These are news items that are somehow related to BBWW. Crime, finances, celebrity, etc.
3. CRIME, FINANCES, CELEBRITY
To analyze this you should first generate a topic model for the buffalo bill data. Play with the number of topics until you find a number that feels about right for the dataset. I am guessing it'll be in the 8-15 range but you'll need to play with it to see what number gives you the best fit. 
4.NUNBER OF TOPICS RANGE OF 8-15
To address the first research question, you'll need to plot topics over time. I would create three models, one for all of the data, one for promotion, and one for reception. What do we learn by doing this?
```{r}
bb.top.terms <- bb.topics %>%
  arrange(desc(beta)) %>% 
  group_by(topic) %>% slice(1:10)

bb.top.terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()
```

```{r}
tidy_bb_new_metadata.freq <- bb %>%
  unnest_tokens(word, text) %>% 
  filter(str_detect(word, "[a-z']$")) %>% 
  anti_join(stop_words)

tidy_bb_new_metadata.freq<- tidy_bb_new_metadata.freq %>% filter(!grepl('[0-9]', word))

```

```{r}
tidy_bb_new_metadata.freq_words <- tidy_bb_new_metadata.freq %>% count(doc_id, word)
```

```{r}

```

For the second, a general topic model of the extraneous articles will be needed. 

```{r}
bb.dtm <- tidy_bb_new_metadata.freq_words %>% 
  count(doc_id, word) %>% 
  cast_dtm(doc_id, word, n)
```

Add code blocks below as necessary.

```{r}
bb.lda <- LDA(bb.dtm, k = 5, control = list(seed = 12345))
bb.lda
```

```{r}
bb.topics <- tidy(bb.lda, matrix = "beta")
head(bb.topics)
```

```{r}
bb.top.terms <- bb.topics %>%
  arrange(desc(beta)) %>% 
  group_by(topic) %>% slice(1:5)

bb.top.terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()
```
This is a mistake pulling schrader, etc.

```{r}
bb.top.terms <- bb.topics %>%
  arrange(desc(beta)) %>% 
  group_by(topic) %>% slice(1:10)

bb.top.terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  scale_y_reordered()
```
#From my original worksheet . . .
The frequency shown in the ggplot bar charts(?) show a great similarity with buffalo being the top word term in three of the charts, second in the fourth, and third in the fifth. In the middle range pull wild, daily, indianapols, exhibition, and world. In the botton tier of the most frequent word are indian, west, daily, rough and riders. The distribution is insightful as to the topics covered in topic modeling this corpus.

```{r warning=FALSE}
library(wordcloud)
topic1 <- bb.topics %>% filter(topic == 2)
wordcloud(topic1$term, topic1$beta, max.words = 100, random.order = FALSE,
    rot.per = 0.3, colors = brewer.pal(6, "Dark2"))
```

In the previous word clowd a plethora of smaller words included for example in green various smaller frequency words from the corpus such as: circus, wabash, goshen, haute, terra, rifle, parade, cleveland, money, exhibit, pharos (curious). My lack of knowleledge make me want to know more about the pharos and if this is connected to the arabs shown in the next cloud.

```{r warning=FALSE}
library(wordcloud)
topic2 <- bb.topics %>% filter(topic == 2)
wordcloud(topic1$term, topic1$beta, max.words = 100, random.order = FALSE,
    rot.per = 0.3, colors = brewer.pal(6, "Dark2"))
```

```{r warning=FALSE}
library(wordcloud)
topic3 <- bb.topics %>% filter(topic == 2)
wordcloud(topic1$term, topic1$beta, max.words = 100, random.order = FALSE,
    rot.per = 0.3, colors = brewer.pal(6, "Dark2"))
```

```{r warning=FALSE}
library(wordcloud)
topic4 <- bb.topics %>% filter(topic == 2)
wordcloud(topic1$term, topic1$beta, max.words = 100, random.order = FALSE,
    rot.per = 0.3, colors = brewer.pal(6, "Dark2"))
```




```{r warning=FALSE}
library(wordcloud)
topic1 <- bb.topics %>% filter(topic == 5)
wordcloud(topic1$term, topic1$beta, max.words = 100, random.order = FALSE,
    rot.per = 0.3, colors = brewer.pal(6, "Dark2"))
```


(@) Finally, when you are all done. Write up your findings. What research question did you focus on and what did you learn? 
My question was general as to what the frequency words showed about Buffalo Bill. From the newspaper articles, what was the print media saying. What terms came to the forefront. What experience was had at the BB shows.

WRITE UP FINDINGS
    The frequency shown in the ggplot bar charts(?) show a great similarity with buffalo being the top word term in three of the charts, second in the fourth, and third in the fifth. In the middle range pull wild, daily, indianapols, exhibition, and world. In the botton tier of the most frequent word are indian, west, daily, rough and riders. The distribution is insightful as to the topics covered in topic modeling this corpus.                           
    In the previous word clowd a plethora of smaller words included for example in green various smaller frequency words from the corpus such as: circus, wabash, goshen, haute, terra, rifle, parade, cleveland, money, exhibit, pharos (curious). My lack of knowleledge make me want to know more about the pharos and if this is connected to the arabs shown in the next cloud.
    In the above word cloud, the largest frequency words shown in gold are buffalo, bill, wild, west and exhibition. 
Some of the midsised frequency in purple are rough, riders, daily, col and people. Other, somewhat smaller terms are in blue are indian, fair, horse,life and interestingly rain. Curious what that term signifies, perhaps rain or shine, the show will go on, maybe. Some of the smaller frequency words included june, war, past, battle and interestingly arabs.
RESEARCH QUESTIONS
WHAT LEARNED
#Wordcloud

#Extraneus material
```{r}
bb.documents <- tidy(ac.lda, matrix = "gamma")
head(bb.documents)
```

#Extraneus material
```{r}
bb.documents %>%  filter(document == "1915_April.txt") %>% arrange(desc(gamma))
```

#Extraneus material
```{r}
topics.by.year <- full_join(bb.documents, bb.metadata, by = join_by(document == filename))
```

#Extraneus material
```{r}
topics.by.year$issue_date <- paste(topics.by.year$month, " ", topics.by.year$year, sep = "")
ggplot(data=topics.by.year, aes(x=issue_date, y=gamma)) + geom_bar(stat="identity") + facet_wrap(~ topic, scales = "free") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
```

#Extraneus material
```{r}
topics.by.year$issue_date <- paste(topics.by.year$month, " ", topics.by.year$year, sep = "")
ggplot(data=topics.by.year, aes(x=issue_date, y=gamma)) + geom_bar(stat="identity") + facet_wrap(~ topic, scales = "free") + theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
```

The chunks at the end were for other experimentation that I may continue with. I do also realize that in some of the questions there may have been a problem of pulling meta data and text files from either different or the wrong corpus juggling between Buffalo Bill, American City and Mind and Body. Although the portions that I was able to run successfully were insightful of the capabilities of Topic Modeling.

**This was the final worksheet for this class. You did it - you learned to code in R! Congrats!**
