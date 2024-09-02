library(tidyverse)
library(caret)

#importing datasets
train <- read.csv("train.csv")

#cleaning dataset
train <- train %>% select(-PassengerId,-Name,-Ticket,-Cabin)
n <- length(train$Cabin)
#for (i in 1:n){
#  if (train$Cabin[i] == ""){
#    train$Cabin[i] = NA
#  }
#}

train
#splitting dataset
splitting <- sample(1:nrow(train),size = 0.8 * nrow(train))
tr <- train[splitting,]
te <- train[-splitting,]

#segmenting dependent and independent variables
y_tr <- factor(tr$Survived)
x_tr <- tr %>% select(-Survived)

logist <- glm(factor(Survived) ~ ., tr, family = binomial)
summary(logist)


x_te <- te %>% select(-Survived)
te_log <- predict(logist,x_te, type = "response")


x_tr
#newstuff-- this branch is new

