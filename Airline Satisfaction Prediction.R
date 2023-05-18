

# Install necessary packages
install.packages("kknn") # For kknn
install.packages("glmnet") # For Ridge Regression
install.packages("doParallel") # required to run massive K-NN analysis
install.packages("class")
install.packages("klaR")
install.packages("e1071")
install.packages("randomForest")
install.packages("leaps")
install.packages("lars")
install.packages("MASS")
install.packages("psych")
install.packages("tidyr")
install.packages("caret")
install.packages("dplyr")
install.packages("rpart")
install.packages("nnet")
library(caret) # Used for SVM and splitting dataset into training/testing
library(fastDummies)
library(tidyr) # required for "drop_na"
library(parallel) # required to run massive K-NN analysis
library(doParallel)
library(class)
library(klaR)
library(e1071)
library(rpart)
library(randomForest)
library(leaps)
library(MASS)
library(lars)
library(psych)
library(caret)
library(dplyr)


# Set working directory
setwd('G:/My Drive/BUMK746/Final Project')

# Import dataset
data <- read.csv("airlinedata.csv")
backup_data <- read.csv("airlinedata.csv")

## Create dummy variables

# Create dummy variables for categorical variables
data <- fastDummies::dummy_cols(data, select_columns = c("Gender", "Customer.Type", "Type.of.Travel", "Class", "satisfaction"))

# Replace spaces with periods in column names
colnames(data) <- gsub(" ", ".", colnames(data))

# Remove unnecessary columns and leave one dummy out for each categorical variable
data$id <- NULL
data$Gender <- NULL
data$Gender_Male <- NULL
data$Customer.Type <- NULL
data$Customer.Type_disloyal.Customer <- NULL
data$Type.of.Travel <- NULL
data$Type.of.Travel_Personal.Travel <- NULL
data$Class <- NULL
data$Class_Eco <- NULL
data$satisfaction <- NULL
data$satisfaction_neutral.or.dissatisfied <- NULL
data$satisfaction <- data$satisfaction_satisfied # Renaming "satisfaction_satisfied" dummy "satisfaction" for simplicity
data$satisfaction_satisfied <- NULL

# Set factor variables
for (i in c("Inflight.wifi.service", "Departure.Arrival.time.convenient", "Ease.of.Online.booking", "Gate.location", "Food.and.drink", "Online.boarding", "Seat.comfort", "Inflight.entertainment", "On.board.service", "Leg.room.service", "Baggage.handling", "Checkin.service", "Inflight.service", "Cleanliness")) {
  data[[i]] <- as.factor(data[[i]])
}

# Check for missing values by getting a sum of all missing values
sum(is.na(data))

# Remove rows with missing values
data <- drop_na(data)

# Partition data into a training portion (80%) and a testing portion (20%)
set.seed(123)
trainIndex <- createDataPartition(data$satisfaction, p = 0.8, list = FALSE, times = 1)
train <- data[trainIndex,]
test <- data[-trainIndex,]



#################

# 1.1) Explore average values for Loyal and Business customers

#################

### LOYAL

# Filter loyal customers
loyal_customers <- data %>% filter(Customer.Type_Loyal.Customer == 1)

# Calculate the average value for specific variables
loyal_table <- loyal_customers %>% summarise(
  avg_Checkin.service = mean(as.numeric(as.character(Checkin.service)), na.rm = TRUE),
  avg_Inflight.wifi.service = mean(as.numeric(as.character(Inflight.wifi.service)), na.rm = TRUE),
  avg_Seat.comfort = mean(as.numeric(as.character(Seat.comfort)), na.rm = TRUE),
  avg_Baggage.handling = mean(as.numeric(as.character(Baggage.handling)), na.rm = TRUE),
  avg_Online.boarding = mean(as.numeric(as.character(Online.boarding)), na.rm = TRUE)
)

### NON-LOYAL

# Filter loyal customers
non_loyal_customers <- data %>% filter(Customer.Type_Loyal.Customer == 0)

# Calculate the average value for specific variables
non_loyal_table <- non_loyal_customers %>% summarise(
  avg_Inflight.wifi.service = mean(as.numeric(as.character(Inflight.wifi.service)), na.rm = TRUE),
  avg_Baggage.handling = mean(as.numeric(as.character(Baggage.handling)), na.rm = TRUE),
  avg_Inflight.service = mean(as.numeric(as.character(Inflight.service)), na.rm = TRUE),
  avg_Checkin.service = mean(as.numeric(as.character(Checkin.service)), na.rm = TRUE),
  avg_On.board.service = mean(as.numeric(as.character(On.board.service)), na.rm = TRUE)
)

### BUSINESS

# Filter business customers
business_customers <- data %>% filter(Type.of.Travel_Business.travel == 1)

# Calculate the average value for specific variables
business_table <- business_customers %>% summarise(
  avg_Checkin.service = mean(as.numeric(as.character(Checkin.service)), na.rm = TRUE),
  avg_Baggage.handling = mean(as.numeric(as.character(Baggage.handling)), na.rm = TRUE),
  avg_Seat.comfort = mean(as.numeric(as.character(Seat.comfort)), na.rm = TRUE),
  avg_Inflight.wifi.service = mean(as.numeric(as.character(Inflight.wifi.service)), na.rm = TRUE),
  avg_Inflight.service = mean(as.numeric(as.character(Inflight.service)), na.rm = TRUE)
)




#################

# 1.2) Correlation Matrix

#################


# Set factor variables
for (i in c("Inflight.wifi.service", "Departure.Arrival.time.convenient", "Ease.of.Online.booking", "Gate.location", "Food.and.drink", "Online.boarding", "Seat.comfort", "Inflight.entertainment", "On.board.service", "Leg.room.service", "Baggage.handling", "Checkin.service", "Inflight.service", "Cleanliness")) {
  data[[i]] <- as.numeric(data[[i]])
}

# Calculate the correlation matrix
cor_matrix <- cor(data)

# Load the 'corrplot' package
#install.packages("corrplot")
library(corrplot)

# Create the correlation plot
corrplot(cor_matrix, method = "color")

M = cor(data)
corrplot(M, method = 'circle', order = 'FPC', type = 'lower', diag = FALSE)



#################

# 2) K-NN

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
#                     WARNING                    # 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

# I commented this out because it takes a very long time to run and we found 
# its performance to be sub par relative to our other models.

#################



# Load libraries
#library(kknn)
#library(class)
#also uses dplyr, caret, parallel, and doParallel

# Normalize only numeric columns
#numeric_columns <- sapply(train, is.numeric)

#train_normalized <- train
#train_normalized[, numeric_columns] <- as.data.frame(lapply(train[, numeric_columns], scale))

#test_normalized <- test
#test_normalized[, numeric_columns] <- as.data.frame(lapply(test[, numeric_columns], scale))

# Train and test features
#train_features <- train_normalized %>% select(-satisfaction)
#test_features <- test_normalized %>% select(-satisfaction)

# Train and test labels
#train_labels <- train$satisfaction
#test_labels <- test$satisfaction

## Cross-validation to find the best k value
#cv_results <- data.frame()

# Load the class library
#library(class)

# Determine the number of cores available on your machine
#num_cores <- detectCores()

# Create a function that performs K-NN with the given k value
#knn_with_k <- function(k) {
#  print("Processing k =", k)
#  pred <- knn(train_features, train_features, cl = train_labels, k = k)
#  cv_result <- data.frame(k = k, accuracy = mean(pred == train_labels))
#  return(cv_result)
#}

# Set up a cluster for parallel processing
#cl <- makeCluster(num_cores)

# Export necessary variables to the cluster
#clusterExport(cl, c("train_features", "train_labels", "knn"))

# Register the parallel backend
#registerDoParallel(cl)

# Run the knn_with_k function in parallel for each k value
#k_values <- c(1:5)
#cv_results <- do.call(rbind, parLapply(cl, k_values, knn_with_k))

# Stop the cluster
#stopCluster(cl)

# Print the results
#print(cv_results)

# Find the best k value
#best_k <- cv_results[which.max(cv_results$accuracy), "k"]
#print(best_k)

# Train the k-NN model with the best k value
#knn_model <- kknn(satisfaction ~ ., train = train_complete, test = test_normalized, k = best_k, scale = FALSE, kernel = "rectangular")

# Make predictions on the test set
#test_pred <- as.integer(fitted(knn_model) > 0.5)

# Evaluate the model's performance
#accuracy <- mean(test_pred == test_labels)
#cat("Test accuracy:", accuracy, "\n")

# Load the caret library if not loaded
#library(caret)

# Generate a confusion matrix for the KNN results
#confusion_matrix <- confusionMatrix(as.factor(test_pred), as.factor(test_labels))

# Print the confusion matrix
#print(confusion_matrix)

#Temp | See how many 1s and 0s there are in data$satisfaction
#table(data$satisfaction)

#################

# 3) Regression

#################



#################

# 3A) Ridge Regression

#################

library(glmnet)

# Prepare input and response variables for training set
x_train <- model.matrix(satisfaction ~ ., data = train)[,-1]
y_train <- train$satisfaction

# Prepare input and response variables for testing set
x_test <- model.matrix(satisfaction ~ ., data = test)[,-1]
y_test <- test$satisfaction

# Set seed for reproducibility
set.seed(123)

# Perform ridge regression with cross-validation to find the optimal lambda value
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 10)

# Fit the ridge regression model using the optimal lambda value
ridge_model <- glmnet(x_train, y_train, alpha = 0, lambda = cv_ridge$lambda.min)

# Predict satisfaction for the test set
predictions <- predict(ridge_model, x_test, type = "response", s = cv_ridge$lambda.min)
predictions_binary <- ifelse(predictions > 0.5, 1, 0)

# Evaluate the model performance
confusion_matrix <- table(Predicted = predictions_binary, Actual = y_test)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(confusion_matrix)
print(paste("Accuracy:", accuracy))

# Calculate precision, recall, and F1 score
precision <- confusion_matrix[2, 2] / (confusion_matrix[2, 2] + confusion_matrix[1, 2])
recall <- confusion_matrix[2, 2] / (confusion_matrix[2, 2] + confusion_matrix[2, 1])
f1_score <- 2 * (precision * recall) / (precision + recall)

# Print precision, recall, and F1 score
print(paste("Precision:", precision))
print(paste("Recall:", recall))
print(paste("F1 Score:", f1_score))

# Get f1_score
ridge_f1 = f1_score

# Extract coefficients from the ridge_model and display them
ridge_coefficients <- coef(ridge_model)
print(ridge_coefficients)

# Convert the sparse matrix to a regular matrix
ridge_coefficients_matrix <- as.matrix(ridge_coefficients)

# Convert the matrix to a data frame
ridge_coefficient_df <- as.data.frame(ridge_coefficients_matrix)

# Drop rows where the value in $s0 is less than 3
ridge_important_vars <- subset(ridge_coefficient_df, s0 >= .05)

# Install and load the pROC library
#install.packages("pROC")
library(pROC)

# Use the roc function to compute the ROC curve
roc_obj <- roc(y_test, predictions)

# Compute the AUC
auc(roc_obj)



#################

# 3B) Logistic Regression

#################



# Load the necessary packages
library(caret)
library(glmnet)

# Fit the logistic regression model
model <- glm(satisfaction ~ ., data = train, family = "binomial")

# Make predictions on the test set
predictions <- predict(model, newdata = test, type = "response")

# Convert probabilities to binary predictions
binary_predictions <- ifelse(predictions > 0.5, "satisfied", "neutral or dissatisfied")

# Evaluate the model
confusion_matrix <- table(test$satisfaction, binary_predictions)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
precision <- confusion_matrix[2, 2] / sum(confusion_matrix[, 2])
recall <- confusion_matrix[2, 2] / sum(confusion_matrix[2, ])
f1_score <- 2 * precision * recall / (precision + recall)

# Print the evaluation metrics
print(confusion_matrix)
cat("Accuracy:", accuracy, "\n")
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")

logistic_f1 = f1_score



#################

# 3C) Forward-stepwise regression

#################



# Load required library
library(lars)
y = data$satisfaction
x = cbind(data$Age, data$Flight.Distance, data$Inflight.wifi.service, data$Departure.Arrival.time.convenient, data$Ease.of.Online.booking, data$Gate.location, data$Food.and.drink, data$Online.boarding, data$Seat.comfort, data$Inflight.entertainment, data$On.board.service, data$Leg.room.service, data$Baggage.handling, data$Checkin.service, data$Inflight.service, data$Cleanliness, data$Departure.Delay.in.Minutes, data$Gender_Female, data$Customer.Type_Loyal.Customer, data$Type.of.Travel_Business.travel, data$Class_Business, data$Class_Eco.Plus)
stepwise_res = lars(x, y, type="stepwise")
print(summary(stepwise_res))
stepwise_res



#################

# 3D) Full model search

#################

#Load required libraries
library(leaps)

# Run full model search
y = data$satisfaction
# Had to preserve the column names by using colname = data$colname for some reason.
x = cbind(data$Age, data$Flight.Distance, data$Inflight.wifi.service, data$Departure.Arrival.time.convenient, data$Ease.of.Online.booking, data$Gate.location, data$Food.and.drink, data$Online.boarding, data$Seat.comfort, data$Inflight.entertainment, data$On.board.service, data$Leg.room.service, data$Baggage.handling, data$Checkin.service, data$Inflight.service, data$Cleanliness, data$Departure.Delay.in.Minutes, data$Gender_Female, data$Customer.Type_Loyal.Customer, data$Type.of.Travel_Business.travel, data$Class_Business, data$Class_Eco.Plus)
full_model_search = leaps(x,y,method="Cp", nbest = 1)
print(full_model_search)

# Find the optimal CP value
optimal_cp = full_model_search$Cp[which.min(full_model_search$Cp)]
print(optimal_cp)

# Find the index number of the optimal CP value
optimal_cp_index = which(full_model_search$Cp == optimal_cp)
print(optimal_cp_index)

# Get a list of True/False values for the row with the optimal CP value
best_model_binary <- full_model_search$which[optimal_cp_index,]

# Convert best_model_binary dataframe to a logical vector
best_model_binary_vector <- as.logical(best_model_binary)

# Make a list with the variable names (the x dataset doesn't have column names so we can't extract them from that)
variable_names <- c("Age", "Flight.Distance", "Inflight.wifi.service", "Departure.Arrival.time.convenient", "Ease.of.Online.booking", "Gate.location", "Food.and.drink", "Online.boarding", "Seat.comfort", "Inflight.entertainment", "On.board.service", "Leg.room.service", "Baggage.handling", "Checkin.service", "Inflight.service", "Cleanliness", "Departure.Delay.in.Minutes", "Gender_Female", "Customer.Type_Loyal.Customer", "Type.of.Travel_Business.travel", "Class_Business", "Class_Eco.Plus")

# Get the optimal list of variables using best_model_binary_vector
optimal_variables <- variable_names[best_model_binary_vector]

# Print the optimal list of variables
print(optimal_variables)





#################

# 4) Reset to prepare for for Decision Tree and Random Forest

#################




set.seed(123)
sample_data <- backup_data 
data = backup_data

#check missing values
colSums(is.na(data))

# Convert NA values to 0 in Arrival.Delay.in.Minutes column
sample_data$Arrival.Delay.in.Minutes[is.na(sample_data$Arrival.Delay.in.Minutes)] <- 0
#data <- drop_na(data)


str(sample_data)

## Create dummy variables
sample_data$Gender <- ifelse(sample_data$Gender=='Female',0,1)
sample_data$Type.of.Travel <- ifelse(sample_data$Type.of.Travel=='Business travel',0,1)
sample_data$Customer.Type <- ifelse(sample_data$Customer.Type=='Loyal Customer', 0,1)   
sample_data$satisfaction <- ifelse(sample_data$satisfaction=='satisfied', 0,1)

#remove irrelevant columns using indexing
new_data <- sample_data[, !(names(sample_data) %in% c("Class",'X','id'))]

# Standardize and demean the numerical columns
new_data$Age <- (new_data$Age - mean(new_data$Age)) / sd(new_data$Age)
new_data$Departure.Delay.in.Minutes <- (new_data$Departure.Delay.in.Minutes - mean(new_data$Departure.Delay.in.Minutes)) / sd(new_data$Departure.Delay.in.Minutes)
new_data$Arrival.Delay.in.Minutes <- (new_data$Arrival.Delay.in.Minutes - mean(new_data$Arrival.Delay.in.Minutes)) / sd(new_data$Arrival.Delay.in.Minutes)
new_data$Flight.Distance <- (new_data$Flight.Distance - mean(new_data$Flight.Distance)) / sd(new_data$Flight.Distance)

str(new_data)
# Replace spaces with periods in column names
colnames(sample_data) <- gsub(" ", ".", colnames(sample_data))

# Set factor variables
for (i in c("satisfaction", 
            "Inflight.wifi.service", 
            "Departure.Arrival.time.convenient", 
            "Ease.of.Online.booking", 
            "Gate.location", "Food.and.drink", 
            "Online.boarding", 
            "Seat.comfort", 
            "Inflight.entertainment", 
            "On.board.service", 
            "Leg.room.service", 
            "Baggage.handling", 
            "Checkin.service",
            "Gender",
            "Customer.Type",
            "Inflight.service", 
            "Cleanliness")) {
  new_data[[i]] <- as.factor(new_data[[i]])
}

# Check for missing values by getting a sum of all missing values
sum(is.na(new_data))


# Partition sample_data into a training portion (80%) and a testing portion (20%)
set.seed(123)
trainIndex <- createDataPartition(new_data$satisfaction, p = 0.8, list = FALSE, times = 1)
train <- new_data[trainIndex,]
test <- new_data[-trainIndex,]






#################

# 4A) Decistion Tree

#################




set.seed(123)  #initialize random seed to make result reproducible
tfit <- rpart(train$satisfaction ~ ., data = train[, -which(names(train) == "satisfaction")], method = 'class', parms = list(split = 'information'), control = rpart.control(cp = 0.0001))


## Actual Values in Columns, predictions in rows
pred1=predict(tfit, train, type="class")
table(predicted= pred1 , Actual_train=  train$satisfaction)

#predict on the testing data2
pred2 = predict(tfit, test, type="class")
table(predicted= pred2, Actual_Test= test$satisfaction)

#calculate precision and recall

myf1score <- function(pred, actual) {
  res = table(pred, actual)
  precision = res[2,2]/(res[2,2]+res[2,1])
  recall = res[2,2]/(res[2,2]+res[1,2])
  f1 = 2/(1/precision+1/recall)
  return(f1)
}
res = table(pred1, train$satisfaction)
precision = res[2,2]/(res[2,2]+res[2,1])
recall = res[2,2]/(res[2,2]+res[1,2])
precision
recall
myf1score(pred1,train$satisfaction)
res = table(pred2, test$satisfaction)
precision = res[2,2]/(res[2,2]+res[2,1])
recall = res[2,2]/(res[2,2]+res[1,2])
precision
recall
myf1score(pred2, test$satisfaction)
#show precision & recall value for training and testing 
# Prune

#how well does it do?
printcp(tfit)
plotcp(tfit)
#lowest xerror is 0.13688, cp= 0.00092666
tfit_pruned = prune(tfit, cp=0.00092666)
plot(tfit_pruned, uniform=TRUE)
text(tfit_pruned)
## Actual Values in Columns, predictions in rows
pred3= predict(tfit_pruned, train, type="class")
table(predicted= pred3, train$satisfaction)
#predict on the testing data2
pred4= predict(tfit_pruned, test, type="class")
table(pred4, test$satisfaction)

res = table(pred3, train$satisfaction)
precision = res[2,2]/(res[2,2]+res[2,1])
recall = res[2,2]/(res[2,2]+res[1,2])
precision
recall
myf1score(pred3,train$satisfaction)

res = table(pred4, test$satisfaction)
precision = res[2,2]/(res[2,2]+res[2,1])
recall = res[2,2]/(res[2,2]+res[1,2])
precision
recall
myf1score(pred4, test$satisfaction)






#################

# 4B) Random Forest

#################





set.seed(123)
rffit <- randomForest(train$satisfaction ~ ., data = train[, -which(names(train) == "satisfaction")], importance = TRUE, na.action = na.omit, ntree =300, mtry=10, max_depth= 10)
plot(rffit)
print(rffit)

# Filter observations with Type.of.Travel equal to 1
rffit_personal <- randomForest(train$satisfaction[train$Type.of.Travel == 1] ~ .,
                               data = train[train$Type.of.Travel == 1, -which(names(train) == "satisfaction")],
                               importance = TRUE, na.action = na.omit, ntree = 300, mtry = 10, max_depth = 10)
plot(rffit_personal)
print(rffit_personal)
varImpPlot(rffit_personal)

# Filter observations with Type.of.Travel equal to 0 (business)
rffit_business <- randomForest(train$satisfaction[train$Type.of.Travel == 0] ~ .,
                               data = train[train$Type.of.Travel == 0, -which(names(train) == "satisfaction")],
                               importance = TRUE, na.action = na.omit, ntree = 300, mtry = 10, max_depth = 10)

plot(rffit_business)
print(rffit_business)
varImpPlot(rffit_business)

#check testing data2
pred6 <- predict(rffit_type0, test[test$Type.of.Travel == 0, ])
table(pred6, test$satisfaction[test$Type.of.Travel==0])
res <- table(pred6, test$satisfaction[test$Type.of.Travel == 0])
myf1score(pred6, test$satisfaction[test$Type.of.Travel == 0])

# non loyal
rffit_nonloyal <- randomForest(train$satisfaction[train$Customer.Type == 1] ~ .,
                               data = train[train$Customer.Type == 1, -which(names(train) == "satisfaction")],
                               importance = TRUE, na.action = na.omit, ntree = 300, mtry = 10, max_depth = 10)
plot(rffit_nonloyal)
print(rffit_nonloyal)
varImpPlot(rffit_nonloyal)

pred6 <- predict(rffit_type1, test[test$Type.of.Travel == 1, ])
table(pred6, test$satisfaction[test$Type.of.Travel==1])
res <- table(pred6, test$satisfaction[test$Type.of.Travel == 1])
myf1score(pred6, test$satisfaction[test$Type.of.Travel == 1])


# Filter observations with Customer.Type equal to 0 (loyal)
rffit_loyal <- randomForest(train$satisfaction[train$Customer.Type == 0] ~ .,
                            data = train[train$Customer.Type == 0, -which(names(train) == "satisfaction")],
                            importance = TRUE, na.action = na.omit, ntree = 300, mtry = 10, max_depth = 10)

plot(rffit_loyal)
print(rffit_loyal)
varImpPlot(rffit_loyal)


plot(rffit)
print(rffit)
varImpPlot(rffit)
pred5= predict(rffit, train)
table(pred5, train$satisfaction)
myf1score(pred5,train$satisfaction)


#check testing data2
pred6= predict(rffit, test)
table(pred6, test$satisfaction)
res = table(pred6, test$satisfaction)

precision = res[2,2]/(res[2,2]+res[2,1])
precision
recall = res[2,2]/(res[2,2]+res[1,2])
recall
myf1score(pred6,test$satisfaction)



##neural network

nn_model <- multinom(satisfaction ~ ., 
                     data = train)
library(caret)
library(pROC)
# Make predictions on the test data
pred <- predict(nn_model, newdata = test, type = "class")
# Calculate accuracy
accuracy <- mean(pred == test$satisfaction)
cat("Accuracy:", accuracy, "\n")

# Create confusion matrix
confusion <- confusionMatrix(pred, test$satisfaction)
cat("Confusion Matrix:\n")
print(confusion)

# Calculate and display classification metrics
classification_metrics <- caret::confusionMatrix(pred, test$satisfaction)
cat("Classification Metrics:\n")
print(classification_metrics)

# Calculate and display ROC curve for each class
roc_curves <- lapply(levels(test$satisfaction), function(class) {
  roc_obj <- roc(response = ifelse(test$satisfaction == class, 1, 0), predictor = ifelse(pred == class, 1, 0))
  plot(roc_obj, main = paste("ROC Curve - Class", class))
  auc <- auc(roc_obj)
  cat(paste("AUC - Class", class, ":", auc, "\n"))
  return(roc_obj)
})
