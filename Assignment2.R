library(tidyverse)
library(ggplot2)
library(dplyr)
library(rpart)
library(pROC)
library(caret)


#DATA EXTRACTION
cardata <- read.csv(file="C:/Users/maria/Desktop/DMASSIGNMENTS/Assignment2/car.txt",header=TRUE,sep=",")

#DATA EXPLORATION

i<<-0;
lapply(cardata %>% select(price:safety),function(x,y=colnames(cardata)){
  i<<-i+1;
  ggplot(cardata)+
    geom_bar(aes(x,fill=shouldBuy),position="dodge")+
    ggtitle(y[[i]])
})

lapply(cardata,function(x){table(x)}) #no typos error

sum(!complete.cases(cardata)) #rows with NA 

cardata <- cardata[complete.cases(cardata),]
#data.samples <- sample(1:nrow(cardata), nrow(cardata) *0.7, replace = FALSE)

#PARTITIONING DATA
data.samples <- createDataPartition(cardata$price,p=0.8,list=FALSE,times=1)

#K FOLD
control = trainControl(method="repeatedcv", number = 10, repeats = 3)


#SPLIT DATA INTO TRAIN AND TEST
train.data <- cardata[data.samples,] 
test.data <- cardata[-data.samples,] %>% select(-shouldBuy)

#MODEL BULDING - Random Forest Tree
#----------------------------------

#RANDOM SEED
set.seed(100)

model_random_forest<-train(shouldBuy ~ .,data=train.data,method="rf",metric="Accuracy",trControl=control)

#PREDICT
pred_random_forest<-predict(model_random_forest,test.data)

#par(mfrow = c(1, 2))


table(pred_random_forest,cardata[-data.samples,]$shouldBuy) #confusion matrix


#ACCURACY, SPECIFICITY, SENSITIVITY, ROC

confusionMatrix(pred_random_forest,factor(cardata[-data.samples,]$shouldBuy))

ggplot(varImp(model_random_forest))
#par(mfrow = c(1, 1))

plot(roc(response=cardata[-data.samples,]$shouldBuy,predictor=factor(pred,ordered = TRUE)))



#MODEL BULDING - Recursive Partioning Tree
#------------------------------------------

#RANDOM SEED
set.seed(100)

model<-rpart(shouldBuy~.,method="class",data=train.data)

#PREDICT
pred<-predict(model,test.data,type="class")

#plotcp(model)

#par(mfrow = c(1, 2))

plot(model, uniform = TRUE, margin = 0, main = "Original Tree")
text(model, use.n = TRUE, all = TRUE, cex = 0.5)

table(pred,cardata[-data.samples,]$shouldBuy) #confusion matrix


#ACCURACY, SPECIFICITY, SENSITIVITY, ROC

confusionMatrix(pred,factor(cardata[-data.samples,]$shouldBuy))


#par(mfrow = c(1, 1))

plot(roc(response=cardata[-data.samples,]$shouldBuy,predictor=factor(pred,ordered = TRUE)))



#ggplot(cardata,aes(x=price,fill=shouldBuy))+geom_bar(position="dodge")


