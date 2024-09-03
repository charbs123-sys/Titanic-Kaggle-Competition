library(tidyverse)
library(caret)
library(gsubfn)
#importing datasets
train <- read.csv("train.csv")

#cleaning dataset
train <- train %>% select(-PassengerId,-Name,-Ticket,-Cabin)
n <- length(train$Cabin)
train_pairwise <- train
train <- train %>% na.omit()


#splitting dataset
splitting <- function(train){
splitting <- sample(1:nrow(train),size = 0.8 * nrow(train))
tr <- train[splitting,]
te <- train[-splitting,]
tr_shuffle <- tr[sample(1:nrow(tr)),]
return(list(train = tr, test = te, train_shuffle = tr_shuffle))
}


#cross validation

cross_val <- function(k,tr_shuffle){
#implementing k-fold CV
fold_size <- floor(nrow(tr_shuffle)/k)
class <- numeric(k)

for(i in 1:k){
  val_indices <- ((i-1) * fold_size + 1): (i * fold_size)
  validation <- tr_shuffle[val_indices,]
  training <- tr_shuffle[-val_indices,]
  
  logist_cv <- glm(factor(Survived) ~ ., data = training, family = binomial())
  
  pred <- ifelse(predict(logist_cv,newdata = validation, type = "response") > 0.5,1,0)
  filtering <- !is.na(pred) & !is.na(validation$Survived)
  pred <- pred[filtering]
  pred <- factor(pred, levels = c(0,1))
  test_pred <- validation$Survived[filtering]
  test_pred <- factor(test_pred,levels = c(0,1))
  conf <- confusionMatrix(pred,test_pred)
  print(conf)

  class[i] <- conf$byClass[11]
}

#final metrics for comparison
class_avg <- mean(class,na.rm = TRUE)
return(class_avg)
}

#10 fold CV with casewise deletion
splits <- splitting(train)
tr_case <- as.data.frame(splits$train_shuffle)
(accuracy_casewise <- cross_val(10,tr_case))



#implement model with highest accuracy -- will implement later with appropriate dataset
tr <- as.data.frame(splits$train)
te <- as.data.frame(splits$test)
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




#---- do pairwise deletion later
#10 fold CV with pairwise deletion (all NA values in predict are removed)
#splits1 <- splitting(train_pairwise)
#tr_pair <- as.data.frame(splits1$train_shuffle)
#tr_pair$Age
#(accuracy_pariwise <- cross_val(5,tr_pair))

