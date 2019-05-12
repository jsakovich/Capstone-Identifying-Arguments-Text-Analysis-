#import dataset

library(readr)
Argument_ID_Raw_Dataset <- read.csv("~/Desktop/Data Science:AI:ML/Capstone Project/Argument ID Raw Dataset.csv", stringsAsFactors = FALSE)
View(Argument_ID_Raw_Dataset)

argID = Argument_ID_Raw_Dataset

#examine the structure of the data

str(argID)

# install tm and SnowballC packages

install.packages("tm")
library(tm)
install.packages("SnowballC")
library(SnowballC)

# create corpus 

corpus = Corpus(VectorSource(argID$Passage))

# data cleaning steps: all to lower case, remove punctuation, remove stop words, stem words in corpus

corpus = tm_map(corpus, tolower)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, c(stopwords("english")))
corpus = tm_map(corpus, stemDocument)

# document term matrix from corpus 

frequencies = DocumentTermMatrix(corpus)

# create sparse matrix

sparse = removeSparseTerms(frequencies, 0.995)
argIDSparse = as.data.frame(as.matrix(sparse))
colnames(argIDSparse) = make.names(colnames(argIDSparse))

# prepare for machine learning

argIDSparse$Argument = argID$Argument

# install caTools package 

install.packages("caTools")
library(caTools)

# set the seed and split corpus for training and testing datasets

set.seed(123)
split = sample.split(argIDSparse$Argument, SplitRatio = 0.7)

trainSparse = subset(argIDSparse, split == TRUE)
testSparse = subset(argIDSparse, split == FALSE)

# install rpart and rpart.plot packages

install.packages("rpart")
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)

# run rpart classification model on training data

argIDCART = rpart(Argument ~ ., data = trainSparse, method = "class")
prp(argIDCART)

# use model to predict for test data

predictCART = predict(argIDCART, newdata = testSparse, type = "class")

# check prediction accuracy - show confusion matrix table

table(testSparse$Argument, predictCART)

table(testSparse$Argument)




# tried a random forest approach but kept getting errors

# install.packages("randomForest")
# library(randomForest)
# set.seed(123)

# install.packages("anchors")
# library(anchors)

# trainSparse$Argument <- gsub('Yes', '1', trainSparse$Argument)
# trainSparse$Argument <- gsub('No', '0', trainSparse$Argument)

# argIDRF = randomForest(Argument ~ ., data = trainSparse)  
