### import dataset and libraries

library(tm)
library(SnowballC)
library(rpart)
library(rpart.plot)
library(caTools)
library(dplyr)

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

### plot variable importance
# I'm stuck here - I can plot the decision tree, but I can't figure out how to plot the variable importance in a way that is helpful to look at
# I was able to extract it with: as.data.frame(argIDCART$variable.importance)
# But I can't figure out how to plot just the variable importance

argPlot <- as.data.frame(argIDCART$variable.importance) # convert variable importance from model to data frame

argPlot 

plot.default(argPlot) # plot data frame with variable importance

argIDCART$frame # examine the frame

argDF <- argIDCART$frame 

plot(argDF [1:2])

### --- prune the tree
pruneFrame <- as.data.frame(printcp(argIDCART))
# - optimal value for the CP parameter
xerror_min <- min(pruneFrame$xerror) 
xerror_min_sd <- pruneFrame$xstd[which(pruneFrame$xerror == min(pruneFrame$xerror))]
limits <- c(xerror_min - xerror_min_sd, xerror_min + xerror_min_sd)
pruneFrame <- filter(pruneFrame, 
                     pruneFrame$xerror <= limits[2] & xerror >= limits[1])
optimal_cp <- pruneFrame$CP[which(pruneFrame$nsplit == min(pruneFrame$nsplit))]

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
# - inspect: I think that we will be observing
# - varying results, for reasons already discussed above ^^
cMatrixPerText$`Test Prep`
cMatrixPerText$`Meditation One`
cMatrixPerText$`Meditation Two`
cMatrixPerText$`Meditation Three`
cMatrixPerText$`Meditation Six`
cMatrixPerText$`8500th Session`



