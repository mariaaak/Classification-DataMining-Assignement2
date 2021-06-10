library(tidyverse)
library(ggplot2)
library(dplyr)
library(rpart)
library(pROC)
library(caret)
library(mlr)



# RANDOM SEED
set.seed(100)

# DATA EXTRACTION
cardata <- read.csv("car.data.csv")
cardata <- map_df(cardata,as.factor)

# DATA EXPLORATION

dim(cardata)
cardata <- map_df(cardata, as.factor)

i <- 0;
lapply(cardata %>% select(price:safety), function(x, y = colnames(cardata)){
    i <<- i + 1;
    ggplot(cardata) +
        geom_bar(aes(x, fill = shouldBuy), position = "dodge") +
        ggtitle(y[i])
})

lapply(cardata, function(x) { table(x) }) #no typos error

sum(!complete.cases(cardata)) #rows with NA 

cardata <- cardata[complete.cases(cardata),]
#data.samples <- sample(1:nrow(cardata), nrow(cardata) *0.7, replace = FALSE)

#PARTITIONING DATA
data.samples <- createDataPartition(cardata$price, p = 0.8, list = FALSE, times = 1)

#SPLIT DATA INTO TRAIN AND TEST
train.data <- cardata[data.samples, ]
test.data <- cardata[-data.samples, ] %>% select(-shouldBuy)

#MODEL BULDING - Random Forest Tree
#----------------------------------

#HYPER TUNING OF PARAMETERS
#tunegrid <- expand.grid(.mtry = (1:15)) 

#USE STRATIFIED K FOLD ?
prop.table(table(cardata[data.samples,]$shouldBuy))
prop.table(table(cardata[-data.samples,]$shouldBuy))

#K FOLD
cartask <- mlr::makeClassifTask(data = mutate_all(cardata,as.factor) , target = "shouldBuy")
tree <- mlr::makeLearner("classif.randomForest")
treeParamSpace <- makeParamSet(
    makeIntegerParam("ntree", lower = 390, upper = 410),
    makeIntegerParam("mtry", lower = 5, upper = 5))

# randSearch <- makeTuneControlRandom(maxit = 150)

gridSearchCV <- makeTuneControlGrid()

# bayesSearch <- makeTuneControlMBO()

cvForTuning <- makeResampleDesc("CV", iters = 10, stratify = TRUE)

tunedTreePars <- tuneParams(tree, task = cartask,
                            resampling = cvForTuning,
                            par.set = treeParamSpace,
                            control = gridSearchCV)

data.samples <- createDataPartition(cardata$price, p = 0.8, list = FALSE, times = 1)

#SPLIT DATA INTO TRAIN AND TEST
train.data <- cardata[data.samples, ]
test.data <- cardata[-data.samples, ] %>% select(-shouldBuy)

rfmodel <- randomForest::randomForest(shouldBuy ~ ., data=train.data, importance=TRUE)

rfmodel <- randomForest::randomForest(shouldBuy ~ ., data=train.data, importance=TRUE, 
                                      ntree=399,
                                      mtry=5)

pred_random_forest <- predict(rfmodel, test.data)
# pred_random_forest.prob <- predict(model_random_forest, test.data, type = "prob")

#par(mfrow = c(1, 2))


table(pred_random_forest, cardata[-data.samples, ]$shouldBuy) #confusion matrix


#ACCURACY, SPECIFICITY, SENSITIVITY, ROC

confusionMatrix(pred_random_forest, factor(cardata[-data.samples, ]$shouldBuy))


plot(roc(response = cardata[-data.samples, ]$shouldBuy, 
         predictor = factor(pred_random_forest, ordered = TRUE)))

text(x = 0, y = 0.1, labels = paste("AUC =", auc(roc(response = cardata[-data.samples, ]$shouldBuy, predictor = factor(pred_random_forest,ordered = TRUE)))))



#MODEL BULDING - Recursive Partitioning Tree
#------------------------------------------


#HYPER TUNING OF PARAMETERS
cartask <- mlr::makeClassifTask(data = mutate_all(cardata,as.factor) , target = "shouldBuy")
tree <- mlr::makeLearner("classif.rpart")
treeParamSpace <- makeParamSet(
    makeIntegerParam("minsplit", lower = 5, upper = 20),
    makeIntegerParam("minbucket", lower = 1, upper = 6),
    makeNumericParam("cp", lower = 0.01, upper = 0.1),
    makeIntegerParam("maxdepth", lower = 5, upper = 30))

# randSearch <- makeTuneControlRandom(maxit = 100)

gridSearchCV <- makeTuneControlGrid()

gridSearchCV <- makeTuneControlMBO()

cvForTuning <- makeResampleDesc("CV", iters = 10, stratify = TRUE)

tunedTreePars <- tuneParams(tree, task = cartask,
                            resampling = cvForTuning,
                            par.set = treeParamSpace,
                            control = gridSearchCV)



# model<-rpart(shouldBuy~.,method="class",data=train.data, control=rpart.control(minsplit=16/,minbucket=6,cp=0.0119,maxdepth=10))
model<-rpart(shouldBuy~.,method="class",
             data=train.data, 
             control=rpart.control(minsplit=17,minbucket=6,cp=0.011,maxdepth=10))

model<-rpart(shouldBuy~.,method="class",
             data=train.data)
# model<-rpart(shouldBuy~.,method="class",data=train.data, control=rpart.control(minsplit=20,minbucket=7,cp=0.01,maxdepth=10))

#PREDICT
pred<-predict(model,test.data,type="class")

pred<-predict(model,train.data,type="class")

plotcp(model)

#par(mfrow = c(1, 2))

plot(model, uniform = TRUE, margin = 0, main = "Original Tree")
text(model, use.n = TRUE, all = TRUE, cex = 0.5)


# table(pred,cardata[-data.samples,]$shouldBuy) #confusion matrix
table(pred,cardata[-data.samples,]$shouldBuy) #confusion matrix

table(pred,cardata[data.samples,]$shouldBuy) #confusion matrix


#ACCURACY, SPECIFICITY, SENSITIVITY, ROC

confusionMatrix(pred,factor(cardata[-data.samples,]$shouldBuy))

confusionMatrix(pred,factor(cardata[data.samples,]$shouldBuy))


#par(mfrow = c(1, 1))

plot(roc(response=cardata[-data.samples,]$shouldBuy,predictor=factor(pred,ordered = TRUE)))
text(x=0, y=0.1, labels = paste("AUC =", auc(roc(response=cardata[-data.samples,]$shouldBuy,predictor=factor(pred,ordered = TRUE)))))

recall(as.factor(as.vector(cardata[-data.samples,]$shouldBuy)), pred)
#ggplot(cardata,aes(x=price,fill=shouldBuy))+geom_bar(position="dodge")
recall <- sensitivity(pred, y, positive="1")
