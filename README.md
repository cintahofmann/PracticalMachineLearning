---
title: "ML Final Project"
author: "C. Hofmann"
date: "October 26, 2017"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Overview

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement â a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

## Libraries

```{r}
library(pacman)
pacman::p_load(caret,randomForest,ggplot2,dplyr,RANN)
```

## Import the data

```{r, results=FALSE}
URL_train <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
URL_validate <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
``` 
Looking at the csv files before, there were lots of NA columns, empty fields, and errors, so I'm setting all of them to 'NA'.

```{r, results=FALSE}
trainData <- read.csv(url(URL_train), na.strings=c("","NA","#DIV/0!"))
validation <- read.csv(url(URL_validate), na.strings=c("","NA","#DIV/0!"))

trainCols <- colnames(trainData)
valCols <- colnames(validation)

trainCols == valCols
```

# Eliminate the missing data 

```{r, results=FALSE}
trainData1 <- trainData[,colSums(is.na(trainData))==0]
valData1 <- validation[,colSums(is.na(validation))==0]

str(trainData1)
str(valData1)
```

There are still unnecessary columns at the beginning of the data set. Taking these out too.

```{r, results=FALSE}
training_clean <- trainData1[,-c(1:7)]
colnames(training_clean)

colnames(valData1)
validation_clean <- valData1[,-c(1:7)]

colnames(training_clean)==colnames(validation_clean)
```

## Split into training and testing set

Now it's time to split the data. I decided to split the training set into a training and a testing set and validate on the original test set.

```{r, results=FALSE}
set.seed(12321)

inTrain = createDataPartition(y=training_clean$classe,p=0.75,list=FALSE)
training = training_clean[inTrain,]
testing = training_clean[-inTrain,]
```

```{r}
nrow(training); nrow(testing)
```

## Preparing data for training

Repeated K-fold cross-validaton (10 folds, 10 repeats) will be performed during model training using the trControl function from the caret package.

```{r, results=FALSE}
set.seed(12321)

ctrl <- trainControl(method="repeatedcv",number=10,repeats=10,
                     classProbs = TRUE, summaryFunction = defaultSummary)
```

## Training model using PLS-DA

I tried using Partial Least Squares Discriminant Analysis first (can be looked up in the caret package information).

```{r, return=FALSE}
pacman::p_load(pls,e1071)
set.seed(12421)

train_PLS <- train(classe ~.,
                   data=training,
                   method="pls",
                   tuneLength=15,
                   trControl=ctrl,
                   metric="ROC",
                   preProc=c("center", "scale"))
```

```{r}
train_PLS
```

After training, I applied it to my test set (the 25% of the original training set) and looked at the confusion matrix.

```{r, return=FALSE}
pred_PLS <- predict(train_PLS, newdata = testing)
```

```{r}
confusionMatrix(data=pred_PLS, testing$classe)
```

Seeing as the accuracy is only about 0.6, I will use a different method next: A Random Forest Model.

## Random Forest 
```{r, return=FALSE}
set.seed(1010101)
trainingRF <- randomForest(classe ~., data=training)
```

```{r}
predRF <- predict(trainingRF, newdata=testing)
confusionMatrix(predRF, testing$classe)
```

Looking at the confusion matrix, the algorithm guessed an overwhelming number of cases correctly. The accuracy is extremely high at 0.9965 and especially great for class A.

Now let's apply the predictor to the validation - or the original test - set.

## Apply to test set

```{r}
valRF <- predict(trainingRF, validation_clean)
valRF
```

## Final thoughts

Based on the good fit and high accuracy of the model, I hope that the out of sample error stays around 0.01. 
