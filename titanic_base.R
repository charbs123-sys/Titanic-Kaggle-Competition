library(tidyverse)
library(caret)
library(MASS)
library(e1071)
#importing datasets
train <- read.csv("train.csv")

#cleaning dataset
train <- train %>% dplyr::select(-PassengerId,-Name,-Ticket,-Cabin)
n <- length(train$Cabin)
train <- train[-which(train$Embarked == ""),]

#mean dataset
train_mean <- train
mean_na <- mean(train_mean$Age,na.rm = TRUE)
train_mean$Age[is.na(train_mean$Age)] <- mean_na

#pairwise dataset
train_na <- train

#regression dataset
train_regress <- train
train_regress$Age <- floor(train_regress$Age)
train_tr <- train[!is.na(train_regress$Age),]
train_te <- train[is.na(train_regress$Age),]

regress_1 <- lm(Age ~ ., data = train_tr) #linear regression is not appropriate (negative values present)

pred_1 <- predict(regress_1,train_te)
train_regress$Age[is.na(train_regress$Age)] <- pred_1

#Last observation carried forward (LOCF) dataset -> averaging observations with similar properties
train_L <- train

train_NA <- train_L[is.na(train_L$Age),]

train_NNA <- train_L %>% group_by(Survived,Sex)
train_NNA <- train_NNA[!is.na(train_NNA$Age),]
train_sum <- train_NNA %>% summarise(means = mean(Age))

row_exist_NNA <- train_sum %>% dplyr::select(Survived,Sex)
row_exists_NA <- train_NA %>% dplyr::select(Survived,Sex)

empt <- c()
for (i in 1:nrow(row_exists_NA)){
  for (j in 1:nrow(row_exist_NNA)){
    bool <- row_exists_NA[i,] %in% row_exist_NNA[j,]
    if (bool[1] & bool[2]){
      empt <- c(empt,train_sum$means[j])
    } } }
train_NA <- train_NA %>% dplyr::select(-Age) %>% mutate(Age = empt)
train_NNA <- train_NNA %>% ungroup()
train_L <- rbind(train_NNA,train_NA)


#casewise dataset
train <- train %>% na.omit()



#splitting dataset
split <- function(train){
splitting <- sample(1:nrow(train),size = 0.8 * nrow(train))
tr <- train[splitting,]
te <- train[-splitting,]
tr_shuffle <- tr[sample(1:nrow(tr)),]
return(list(train = tr, test = te, shuffle = tr_shuffle))
}

#different ML models to implement
func_binom <- function(training,type,validation) {
  if (type == "b") {
    ML_output <- glm(factor(Survived) ~ ., data = training,family = binomial())
    pred <- ifelse(predict(ML_output,newdata = validation, type = "response") > 0.5,1,0)
  } else if (type == "l") {
    ML_output <- lda(factor(Survived) ~., data = training)
    pred <- predict(ML_output, newdata = validation)$class
  } else if (type == "q") {
    ML_output <- qda(factor(Survived) ~., data = training)
    pred <- predict(ML_output, newdata = validation)$class
  } else if (type == "B") {
    ML_output <- naiveBayes(factor(Survived) ~., data = training)
    pred <- predict(ML_output,newdata = validation)
  }
  return(pred)
}


cross_val <- function(k,tr_shuffle,type){
  #implementing k-fold CV
  fold_size <- floor(nrow(tr_shuffle)/k)
  class <- numeric(k)
  for(i in 1:k){
    val_indices <- ((i-1) * fold_size + 1): (i * fold_size)
    validation <- tr_shuffle[val_indices,]
    training <- tr_shuffle[-val_indices,]
    
    logist_cv <- func_binom(training,type,validation)
    
    pred <- factor(logist_cv)
    test_pred <- factor(validation$Survived)
    conf <- confusionMatrix(pred,test_pred)
    class[i] <- conf$byClass[11]
  }
  
  #final metrics for comparison
  class_avg <- mean(class)
  return(class_avg)
}


#casewise deletion
splitted <- split(train)
shuffle <- as.data.frame(splitted$shuffle)
(accuracy_casewise <- cross_val(10,shuffle,"b"))

#pairwise deletion
split_pair <- split(train_na)
shuffle_pair <- as.data.frame(split_pair$shuffle)
(accuracy_pair <- cross_val(10,shuffle_pair,"b"))

#mean imputation
split_mean <- split(train_mean)
shuffle_mean <- as.data.frame(split_mean$shuffle)
(accuracy_mean <- cross_val(10,shuffle_mean,"b"))

#regression imputation
split_regress <- split(train_regress)
shuffle_regress <- as.data.frame(split_regress$shuffle)
(accuracy_regress <- cross_val(10,shuffle_regress,"b"))

#LOCF 
split_LOCF <- split(train_L)
shuffle_LOCF <- as.data.frame(split_LOCF$shuffle)
(accuracy_LOCF <- cross_val(10,shuffle_LOCF,"b"))




#retrain model with highest accuracy


#implement model with highest accuracy
#y_tr <- factor(tr$Survived)
#x_tr <- tr %>% dplyr::select(-Survived)

#logist <- glm(factor(Survived) ~ ., tr, family = binomial)
#summary(logist)


#x_te <- te %>% dplyr::select(-Survived)
#te_log <- predict(logist,x_te, type = "response")



#determining preformance against test data
#outcomes <- factor(ifelse(te_log > 0.5,1,0))
#y_te <- factor(te$Survived)
#(conf <- confusionMatrix(y_te,outcomes))


