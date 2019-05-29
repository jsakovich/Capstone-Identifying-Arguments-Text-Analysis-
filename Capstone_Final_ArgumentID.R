###  libraries

library(tm)
library(SnowballC)
library(rpart)
library(rpart.plot)
library(caTools)
library(dplyr)
library(tidyverse)

### import dataset

library(readr)
Argument_ID_Raw_Dataset <- 
  read.csv("Argument ID Raw Dataset.csv", 
           stringsAsFactors = FALSE)

argID = Argument_ID_Raw_Dataset
rm(Argument_ID_Raw_Dataset)

#### examine the structure of the data

str(argID)

### create corpus 

corpus = Corpus(VectorSource(argID$Passage))

### --- data cleaning steps: all to lower case, 
### --- remove punctuation, remove stop words, 
### --- stem words in corpus

corpus = tm_map(corpus, tolower)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, c(stopwords("english")))
corpus = tm_map(corpus, stemDocument)

### document term matrix from corpus 

frequencies = DocumentTermMatrix(corpus)

### remove sparse terms

sparse = removeSparseTerms(frequencies, 0.995)
argIDSparse = as.data.frame(as.matrix(sparse))
colnames(argIDSparse) = make.names(colnames(argIDSparse))

### prepare for machine learning

argIDSparse$Argument = argID$Argument

### --- set the seed and split corpus for training and testing datasets
set.seed(123)
split = sample.split(argIDSparse$Argument, SplitRatio = 0.7)
trainSparse = subset(argIDSparse, split == TRUE)
testSparse = subset(argIDSparse, split == FALSE)

### run rpart classification model on training data

argIDCART = rpart(Argument ~ ., 
                  data = trainSparse, 
                  method = "class")

### --- inspect the solution
printcp(argIDCART) # - display cp table
summary(argIDCART)	
plot(argIDCART)	# - plot the tree
text(argIDCART)	# - add labels
prp(argIDCART) # - nicer plot

### plot variable importance using ggplot()

df <- data.frame(imp = argIDCART$variable.importance[1:9]) # convert variable importance to data frame
df2 <- df %>% 
  tibble::rownames_to_column() %>% 
  dplyr::rename("variable" = rowname) %>% 
  dplyr::arrange(imp) %>%
  dplyr::mutate(variable = forcats::fct_inorder(variable))
ggplot2::ggplot(df2) +
  geom_col(aes(x = variable, y = imp),
           col = "black", show.legend = F) +
  coord_flip() +
  scale_fill_grey() +
  theme_bw()

ggplot2::ggplot(df2) +
  geom_segment(aes(x = variable, y = 0, xend = variable, yend = imp), 
               size = 1.2, alpha = 0.6) +
  geom_point(aes(x = variable, y = imp, col = variable), 
             size = 3.2, show.legend = F) +
  coord_flip() +
  theme_bw()

### --- prune the tree
printcp(argIDCART)
# - optimal value for the CP parameter
# - note: it is found where xerror has its minimum:
optCP <- argIDCART$cptable[which.min(argIDCART$cptable[,"xerror"]),"CP"]
argIDCART_pr <- prune(argIDCART, cp = optCP)
prp(argIDCART_pr)

### plot variable importance for pruned tree using ggplot()

pr_df <- data.frame(imp = argIDCART_pr$variable.importance[1:9]) # convert variable importance to data frame
pr_plot_df <- pr_df %>% 
  tibble::rownames_to_column() %>% 
  dplyr::rename("variable" = rowname) %>% 
  dplyr::arrange(imp) %>%
  dplyr::mutate(variable = forcats::fct_inorder(variable))
ggplot2::ggplot(pr_plot_df) +
  geom_col(aes(x = variable, y = imp),
           col = "black", show.legend = F) +
  coord_flip() +
  scale_fill_grey() +
  theme_bw()

ggplot2::ggplot(pr_plot_df) +
  geom_segment(aes(x = variable, y = 0, xend = variable, yend = imp), 
               size = 1.2, alpha = 0.6) +
  geom_point(aes(x = variable, y = imp, col = variable), 
             size = 3.2, show.legend = F) +
  coord_flip() +
  theme_bw()

# use model to predict for test data

predictCART = predict(argIDCART, newdata = testSparse, type = "class")

### --- use model to predict for test data
predictCART = predict(argIDCART, 
                      newdata = testSparse, 
                      type = "class")
# check prediction accuracy - show confusion matrix table
cMatrix <- table(testSparse$Argument, predictCART)
cMatrix
accuracy <- (cMatrix[1, 1] + cMatrix[2, 2])/sum(cMatrix)
accuracy

### use prunned tree on test data:
predictCART_pr = predict(argIDCART_pr, 
                         newdata = testSparse, 
                         type = "class")
# check prediction accuracy - show confusion matrix table
cMatrix <- table(testSparse$Argument, predictCART_pr)
cMatrix
accuracy <- (cMatrix[1, 1] + cMatrix[2, 2])/sum(cMatrix)
accuracy

### --- Check how accuracy across the
### --- different types of `Text` in the argID data.frame:
argID$Split <- split
argID_checkTextAcc <- filter(argID, !split)
argID_checkTextAcc$Prediction <- predictCART_pr
cMatrixPerText <- lapply(unique(argID_checkTextAcc$Text), 
                         function(x) {
                           d <- filter(argID_checkTextAcc, 
                                       Text == x)
                           return(table(d$Argument, d$Prediction))
                         })
names(cMatrixPerText) <- unique(argID_checkTextAcc$Text)
# - inspect: 
cMatrixPerText$`Test Prep`
cMatrixPerText$`Meditation One`
cMatrixPerText$`Meditation Two`
cMatrixPerText$`Meditation Three`
cMatrixPerText$`Meditation Six`
cMatrixPerText$`8500th Session`



