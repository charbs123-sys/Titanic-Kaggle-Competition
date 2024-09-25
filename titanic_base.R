library(tidyverse)
library(caret)
library(MASS)
library(e1071)
library(randomForest)
library(openxlsx)
library(gbm)
#importing datasets
train <- read.csv("train.csv")

#cleaning dataset
train <- train %>% dplyr::select(-PassengerId,-Ticket,-Cabin)
n <- length(train$Cabin)
train <- train[-which(train$Embarked == ""),]
train <- train %>% mutate(family = SibSp + Parch + 1) %>% dplyr::select(-SibSp,-Parch)

#adding column for class of individual
n <- length(train$Name)
empt <- c()
for (i in 1:n) {
  if (grepl("Mr.",train$Name[i])) {
    empt <- c(empt,"Mr")
  } else if (grepl("Miss",train$Name[i])) {
    empt <- c(empt,"Miss")
  } else if (grepl("Mrs",train$Name[i])) {
    empt <- c(empt,"Mrs")
  } else if (grepl("Master",train$Name[i])) {
    empt <- c(empt,"Master")
  } else {
    empt <- c(empt,"Rare")
  }
}
train <- train %>% mutate(title = factor(empt)) %>% dplyr::select(-Name)




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
  } else if (type == "r") {
    ML_output <- randomForest(factor(Survived) ~ ., data = training, importance = TRUE, proximity = TRUE, ntree = 2000)
    pred <- predict(ML_output,newdata = validation)
  } else if (type == "gbm") {
    
    training$Sex <- factor(training$Sex)
    training$Embarked <- as.factor(training$Embarked)
    training$title <- as.factor(training$title)
    
    validation$Sex <- factor(validation$Sex)
    validation$Embarked <- as.factor(validation$Embarked)
    validation$title <- as.factor(validation$title)
    
    ML_output <- gbm(Survived ~ ., data = training, distribution = "bernoulli", 
                     shrinkage = 0.01, verbose = F, n.trees = 10000, cv.folds = 10)
    pred <- ifelse(predict(ML_output,newdata = validation, type = "response") > 0.5,1,0)
  }
  return(pred)
}


factoring <- function(shuffle_LOCF) {
  shuffle_LOCF$Sex <- factor(shuffle_LOCF$Sex)
  shuffle_LOCF$Embarked <- as.factor(shuffle_LOCF$Embarked)
  shuffle_LOCF$title <- as.factor(shuffle_LOCF$title)
  return(shuffle_LOCF)
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
(accuracy_casewise <- cross_val(10,shuffle,"r"))

#pairwise deletion
split_pair <- split(train_na)
shuffle_pair <- as.data.frame(split_pair$shuffle)
#(accuracy_pair <- cross_val(10,shuffle_pair,"r")) #has NA values do not use with RF

#mean imputation
split_mean <- split(train_mean)
shuffle_mean <- as.data.frame(split_mean$shuffle)
(accuracy_mean <- cross_val(10,shuffle_mean,"r"))

#regression imputation
split_regress <- split(train_regress)
shuffle_regress <- as.data.frame(split_regress$shuffle)
(accuracy_regress <- cross_val(10,shuffle_regress,"r"))

#LOCF 
split_LOCF <- split(train_L)
shuffle_LOCF <- as.data.frame(split_LOCF$shuffle)
(accuracy_LOCF <- cross_val(10,shuffle_LOCF,"r"))



#retrain model with highest accuracy

final_model <- randomForest(factor(Survived) ~ ., data = train_L, importance = TRUE, proximity = TRUE, ntree = 1500)
#train_L <- factoring(train_L)
#final_model <- gbm(Survived ~ ., data = train_L, distribution = "bernoulli", 
#                   shrinkage = 0.01, verbose = T, n.trees = 20000, cv.folds = 10)



#control <- trainControl(method = "repeatedcv", number = 10, repeats = 3, search = 'grid')
#rf_random <- train(factor(Survived) ~ ., data = train_L, method = "rf", metric = "Accuracy", 
#                   tuneLength = 15, trControl = control)


#plot(rf_random)

#test
test_final <- read.csv("test.csv")

pass_id <- test_final$PassengerId
test_final <- test_final %>% dplyr::select(-PassengerId,-Ticket,-Cabin)
n <- length(test_final$Cabin)
test_final <- test_final %>% mutate(family = SibSp + Parch + 1) %>% dplyr::select(-SibSp,-Parch)

n <- length(test_final$Name)
empt <- c()
for (i in 1:n) {
  if (grepl("Mr.",test_final$Name[i])) {
    empt <- c(empt,"Mr")
  } else if (grepl("Miss",test_final$Name[i])) {
    empt <- c(empt,"Miss")
  } else if (grepl("Mrs",test_final$Name[i])) {
    empt <- c(empt,"Mrs")
  } else if (grepl("Master",test_final$Name[i])) {
    empt <- c(empt,"Master")
  } else {
    empt <- c(empt,"Rare")
  }
}
test_final <- test_final %>% mutate(title = factor(empt)) %>% dplyr::select(-Name)

mean_na <- mean(test_final$Age,na.rm = TRUE)
test_final$Age[is.na(test_final$Age)] <- mean_na

pred_test <- predict(final_model,newdata = test_final)

#pred_test <- ifelse(predict(final_model,newdata = test_final, type = 'response') > 0.5,1,0)

names(pred_test) <- FALSE
final_tibble <- data.frame(PassengerId = pass_id,Survived = pred_test)
final_tibble[is.na(final_tibble$Survived),2] <- 0

temp_file <- write.xlsx(final_tibble, file = "C:\\Users\\User\\Documents\\kag_tables")

