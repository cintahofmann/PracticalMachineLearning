
#locate data
URL_train <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
URL_validate <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

#Import data
trainData <- read.csv(url(URL_train), na.strings=c("","NA","#DIV/0!"))
validation <- read.csv(url(URL_validate), na.strings=c("","NA","#DIV/0!"))

trainCols <- colnames(trainData)
valCols <- colnames(validation)

trainCols == valCols

#libraries
library(pacman)
pacman::p_load(caret,randomForest,ggplot2,dplyr,RANN)

#eliminate missing data 
trainData1 <- trainData[,colSums(is.na(trainData))==0]
valData1 <- validation[,colSums(is.na(validation))==0]

#delete unnecessary columns

training_clean <- trainData1[,-c(1:7)]
colnames(training_clean)

colnames(valData1)
validation_clean <- valData1[,-c(1:7)]

colnames(training_clean)==colnames(validation_clean)

#Split into training and testing set
set.seed(12321)

inTrain = createDataPartition(y=training_clean$classe,p=0.75,list=FALSE)
training = training_clean[inTrain,]
testing = training_clean[-inTrain,]

nrow(training); nrow(testing)

#Parameter tuning
ctrl <- trainControl(method="repeatedcv",number=10,repeats=10,
                     classProbs = TRUE, summaryFunction = defaultSummary)

#PLSDA
pacman::p_load(pls,e1071)
set.seed(12421)

train_PLS <- train(classe ~.,
                   data=training,
                   method="pls",
                   tuneLength=15,
                   trControl=ctrl,
                   metric="ROC",
                   preProc=c("center", "scale"))

train_PLS
pred_PLS <- predict(train_PLS, newdata = testing)
PLSc <- confusionMatrix(data=pred_PLS, testing$classe)
PLSc$table


  #Accuracy: 0.6083

#GLM
set.seed(1010101)
trainingRF <- randomForest(classe ~., data=training)

#RF
rf <- train(classe ~., data=training, method="rf", trControl = ctrl)
p <- predict(rf, testing)
confusionMatrix(p, testing$classe)

trainingRF
summary(trainingRF)
trainingRF$forest

predRF <- predict(trainingRF, testing)
confusionMatrix(predRF, testing$classe)

valRF <- predict(trainingRF, validation_clean)
valRF

plot(valRF, log="y")
MDSplot(trainingRF, testing$classe)
getTree(randomForest(classe ~., data=training),3,labelVar = TRUE)

library(party)
parytRF <- cforest(classe~., data=training, control = cforest_unbiased(ntree=50))

parytRF

predict(parytRF, testing)
confusionMatrix(parytRF, testing$classe)

