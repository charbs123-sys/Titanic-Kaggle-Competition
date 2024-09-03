library(tidyverse)
library(caret)

#importing datasets
train <- read.csv("train.csv")

#cleaning dataset
train <- train %>% select(-PassengerId,-Name,-Ticket,-Cabin)
n <- length(train$Cabin)

train <- train %>% na.omit()
train <- train[-which(train$Embarked == ""),]

#splitting dataset
splitting <- sample(1:nrow(train),size = 0.8 * nrow(train))
tr <- train[splitting,]
te <- train[-splitting,]
tr_shuffle <- tr[sample(1:nrow(tr)),]

#cross validation

cross_val <- function(k,tr_shuffle){
  #implementing k-fold CV
  k <- 10
  fold_size <- floor(nrow(tr_shuffle)/k)
  class <- numeric(k)
  i <- 9
  for(i in 1:k){
    val_indices <- ((i-1) * fold_size + 1): (i * fold_size)
    validation <- tr_shuffle[val_indices,]
    training <- tr_shuffle[-val_indices,]
    
    logist_cv <- glm(factor(Survived) ~ ., data = training,family = binomial())
    
    pred <- ifelse(predict(logist_cv,newdata = validation, type = "response") > 0.5,1,0)
    pred <- factor(pred)
    test_pred <- factor(validation$Survived)
    conf <- confusionMatrix(pred,test_pred)
    class[i] <- conf$byClass[11]
  }
  
  #final metrics for comparison
  class_avg <- mean(class)
  return(class_avg)
}

accuracy_casewise <- cross_val(10,tr_shuffle)


#implement model with highest accuracy
y_tr <- factor(tr$Survived)
x_tr <- tr %>% select(-Survived)

logist <- glm(factor(Survived) ~ ., tr, family = binomial)
summary(logist)


x_te <- te %>% select(-Survived)
te_log <- predict(logist,x_te, type = "response")



#determining preformance against test data
outcomes <- factor(ifelse(te_log > 0.5,1,0))
y_te <- factor(te$Survived)
(conf <- confusionMatrix(y_te,outcomes))

