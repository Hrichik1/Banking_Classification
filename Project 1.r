library(dplyr)
colnames(new_train)
colnames(new_test)

#Extracting clmn 

unique(new_train$job)
unique(new_train$marital)
unique(new_train$education)
unique(new_train$default)
unique(new_train$housing)
unique(new_train$loan)
unique(new_train$contact)
unique(new_train$month)
unique(new_train$day_of_week)
unique(new_train$poutcome)

# converting categorical value to numeric value using levels andlabels

new_train$job <- factor(new_train$job, 
                        levels = c("blue-collar","entrepreneur","retired","admin.","student","services","technician","selfemployed","management","unemployed","unknown","housemaid"), 
                        labels = c(1,2,3,4,5,6,7,8,9,10,11,12))
new_train$marital <- factor(new_train$marital, 
                            levels = c("married","divorced","single","unknown"), 
                            labels = c(1,2,3,4))
new_train$education <- factor(new_train$education, 
                              levels=c("basic.9y","university.degree","basic.4y","high.school","professional.course","unknown",
                                       "basic.6y","illiterate"), 
                              labels = c(1,2,3,4,5,6,7,8))
new_train$default <- factor(new_train$default, 
                            levels= c("unknown","no","yes"), 
                            labels = c(1,2,3))
new_train$housing <- factor(new_train$housing, 
                            levels = c("yes","no","unknown"), 
                            labels = c(1,2,3))
new_train$loan <- factor(new_train$loan, 
                         levels = c("yes","no","unknown"), 
                         labels = c(1,2,3))
new_train$contact <- factor(new_train$contact, 
                            levels = c("cellular","telephone"), 
                            labels = c(0,1))
new_train$month <- factor(new_train$month, 
                          levels = c("mar","apr","may","jun","jul","aug","sep","oct","nov","dec"), 
                          labels = c(1,2,3,4,5,6,7,8,9,10))
new_train$day_of_week <- factor(new_train$day_of_week, 
                                levels = c("mon","tue","wed","thu","fri"), 
                                labels = c(1,2,3,4,5))
new_train$poutcome <- factor(new_train$poutcome, 
                             levels = c("success","failure","nonexistent"), 
                             labels = c(1,2,3))

total_instances <- nrow(new_train)
train_indices <- sample(total_instances, 5000)
train_data <- new_train[train_indices, ]
remaining_indices <- setdiff(1:total_instances, train_indices)
test_indices <- sample(remaining_indices, 2000)
test_data <- new_train[test_indices, ]

train_data <- subset(train_data, select = -c(pdays, previous))
test_data <- subset(test_data, select = -c(pdays, previous))

total_instances <- nrow(new_train)


train_data<- rename(train_data, "Class"= y)
test_data<- rename(test_data, "Class"= y)


train_data$Class <- factor(train_data$Class,
                           levels = c("yes","no"),
                           labels = c(1,0))
test_data$Class <- factor(test_data$Class,
                          levels = c("yes","no"),
                          labels = c(1,0))


attributes_with_zero_sd <- names(train_data)[apply(train_data, 2, sd) == 0]
train_data <- train_data[, !colnames(train_data) %in% attributes_with_zero_sd]




train_data[] <- lapply(train_data, function(x) if(is.factor(x)) as.numeric(as.character(x)) else x)



attributes_with_zero_sd <- names(train_data)[apply(train_data, 2, sd) == 0]
train_data <- train_data[, !colnames(train_data) %in% attributes_with_zero_sd]


correlation_matrix <- cor(train_data, method = "pearson")


correlation_threshold <- 0.04

# Compute the correlation matrix
correlation_matrix <- cor(train_data, method = "pearson")

# Extract correlations with the target variable
correlation_with_target <- correlation_matrix['Class', ]

# Filter attributes based on correlation threshold
attributes_to_keep <- names(correlation_with_target[abs(correlation_with_target) > correlation_threshold])
attributes_to_keep <- attributes_to_keep[!is.na(attributes_to_keep)]

# Subset correlation matrix and train data based on selected attributes
correlation_matrix_subset <- correlation_matrix[attributes_to_keep, attributes_to_keep]





train_attributes_with_zero_sd <- names(train_data)[apply(train_data, 2, sd) == 0]
train_data <- train_data[, !colnames(train_data) %in% train_attributes_with_zero_sd]
train_data_filtered <- train_data[, c(attributes_to_keep)]
head(train_data_filtered)


test_attributes_with_zero_sd <- names(test_data)[apply(test_data, 2, sd) == 0]
test_data <- test_data[, !colnames(test_data) %in% test_attributes_with_zero_sd]
test_data_filtered <- test_data[, c(attributes_to_keep)]
head(test_data_filtered)


write.csv(train_data_filtered, file = "Train_KNN.csv")
write.csv(test_data_filtered, file = "Test_KNN.csv")

train <- train_data_filtered
test <- test_data_filtered


#Above all are the preprocessing and corelation matrix to 
#understand the dataset relationship 

# The main KNN start from here :

euclidean_distance <- function(a1,a2,a3,a4,a5,a6,a7,a8, b1,b2,b3,b4,b5,b6,b7,b8) 
{
  sqrt((a1 - b1)^2 + (a2 - b2)^2 + (a3 - b3)^2 + (a4 - b4)^2 + (a5 - b5)^2 + (a6 - b6)^2 + (a7 - b7)^2 + (a8 - b8)^2)
                                                                                             
}

manhattan_distance <- function(a1,a2,a3,a4,a5,a6,a7,a8, b1,b2,b3,b4,b5,b6,b7,b8) 
{
  abs(a1 - b1) + abs(a2 - b2) + abs(a3 - b3) + abs(a4 - b4) + abs(a5 - b5) + abs(a6 - b6) + abs(a7 -
                                                                                                  b7) + abs(a8 - b8))
}

max_dimensional_distance <- function(a1,a2,a3,a4,a5,a6,a7,a8, b1,b2,b3,b4,b5,b6,b7,b8)
{
  max(abs(a1 - b1), abs(a2 - b2) ,abs(a3 - b3), abs(a4 - b4), abs(a5 - b5), abs(a6 - b6), abs(a7 -
                                                                                                b7), abs(a8 - b8))
}



knn_classification <- function(test, train, k, distance_function) {
  predictions <- character(nrow(test))
  
  for (i in 1:nrow(test)) {
    distances <- numeric(nrow(train))
    
    for (j in 1:nrow(train)) {
      distances[j] <- distance_function(test$age[i], test$marital[i], test$default[i], test$contact[i], 
                                        test$duration[i], test$campaign[i], test$poutcome[i], test$Class[i],
                                        test$age[i], train$marital[j], train$default[j], train$contact[j], 
                                        train$duration[j], train$campaign[j], train$poutcome[i], train$Class[j])
    }
    k_indices <- order(distances)[1:k]
    k_class <- train_data$Class[k_indices]
    predictions[i] <- names(which.max(table(k_class)))
  }
  return(predictions)
}


k_values <- c(1,3,5,7,9,11,13,15,17,20)


train <- train[sample(nrow(train)), ]
test <- test[sample(nrow(test)), ]


euAccuracy <- numeric(length(k_values))
m =1
for (k in k_values) {
  
  predictions <- knn_classification(test, train, k, euclidean_distance)
  accuracy <- sum(predictions == test$Class) / nrow(test)
  euAccuracy[m] <- accuracy
  recall <- 0 
  if (sum(test$Class == 1) > 0) {
    recall <- sum(predictions == 1 & test$Class == 1) / sum(test$Class == 1)
  }
  cat("k:", k, "\n")
  cat("Accuracy:", accuracy * 100, "%\n")
  cat("Recall:", recall * 100, "%\n")
  cat("----------------------------\n")
  m=m+1 
}


manAccuracy <- numeric(length(k_values))
i =1
for (k in k_values) {
  predictions <- knn_classification(test, train, k, manhattan_distance)
  accuracy <- sum(predictions == test$Class) / nrow(test)
  manAccuracy[i] <- accuracy
  recall <- sum(predictions == 1 & test$Class == 1) /
    sum(test$Class == 1)
  
  cat("k:", k, "\n")
  cat("Accuracy:", accuracy * 100, "%\n")
  cat("Recall:", recall * 100, "%\n")
  cat("----------------------------\n")
  i=i+1
}

