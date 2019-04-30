library(readr)
Argument_ID_Raw_Dataset <- read.csv("~/Desktop/Data Science:AI:ML/Capstone Project/Argument ID Raw Dataset.csv", stringsAsFactors = FALSE)
View(Argument_ID_Raw_Dataset)

argID = Argument_ID_Raw_Dataset

str(argID)

install.packages("tm")
library(tm)
install.packages("SnowballC")
library(SnowballC)

corpus = Corpus(VectorSource(argID$Passage))

corpus = tm_map(corpus, tolower)
corpus = tm_map(corpus, removePunctuation)
corpus = tm_map(corpus, removeWords, c(stopwords("english")))
corpus = tm_map(corpus, stemDocument)

frequencies = DocumentTermMatrix(corpus)

sparse = removeSparseTerms(frequencies, 0.995)
argIDSparse = as.data.frame(as.matrix(sparse))
colnames(argIDSparse) = make.names(colnames(argIDSparse))

argIDSparse$Argument = argID$Argument

install.packages("caTools")
library(caTools)

set.seed(123)
split = sample.split(argIDSparse$Argument, SplitRatio = 0.7)

trainSparse = subset(argIDSparse, split == TRUE)
testSparse = subset(argIDSparse, split == FALSE)

install.packages("rpart")
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)

argIDCART = rpart(Argument ~ ., data = trainSparse, method = "class")
prp(argIDCART)

predictCART = predict(argIDCART, newdata = testSparse, type = "class")

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
