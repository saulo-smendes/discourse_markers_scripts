
# ANEXO D1.3 Leaps and bounds e LDA
library("readxl")
library("MASS")
library("caret")
library("cluster")
#library("pgirmess")


# Install and load the leaps package
if (!requireNamespace("leaps", quietly = TRUE)) {
  install.packages("leaps")
}
library("leaps")


# Load the data from a csv file in "C:\Users\Saulo Mendes Santos\OneDrive\Documents\2. LETRAS\0. Doutorado\0. Recherche\1.5. Classification"
all <- read.csv("C:/Users/Saulo Mendes Santos/OneDrive/Documents/2. LETRAS/0. Doutorado/0. Recherche/1.5. Classification/results_dm_vfinal_standardized_withf0coeff_20231205.csv")

# Extract features and target variable
# Features are contained in column 6 to 40
features <- all[, 6:40]

# Exclude columns from 2 to 4
data <- features[, -c(2:5)]

# Define the target variable
target <- as.factor(all$iu)


# Get names of selected features
feature_names <- names(features)

# Check types of features
sapply(data, class)

# Imputation of missing values in features
# Replace missing values with the mean of the column
data <- apply(data, 2, function(x) replace(x, is.na(x), mean(x, na.rm = TRUE)))

# Create a list to store the classifiers
classifiers <- list()

# Function to train a classifier and return selected features
# Function to train a classifier and return selected features
train_classifier <- function(data, target, class_label) {
  # Create a binary target variable: class vs. OTHERS
  binary_target <- ifelse(target == class_label, class_label, "OTHERS")
  binary_target <- as.factor(binary_target)
  
  # Perform feature selection using leaps and train LDA
  REGS <- regsubsets(data, binary_target, int = TRUE, names = feature_names, nbest = 60, nvmax = 30, really.big = TRUE)
  
  # Find the index with the minimum Cp value
  min_cp_index <- which.min(summary(REGS)$cp)
  
  # Get the selected features based on the minimum Cp index
  selected_features <- names(which(summary(REGS)$which[min_cp_index, ]))
  
  LDA_model <- lda(binary_target ~ ., data = data[, c("Tag", selected_features)])
  
  return(list(model = LDA_model, features = selected_features))
}
# Function to find the classifier with the least features
find_min_features_classifier <- function(classifiers) {
  num_features <- sapply(classifiers, function(x) length(x$features))
  min_features_index <- which.min(num_features)
  return(min_features_index)
}

# Function to perform sequential classification
sequential_classification <- function(data, target, classifiers) {
  predicted_labels <- rep(NA, length(target))
  
  while (length(unique(predicted_labels[is.na(predicted_labels)])) > 1) {
    min_features_classifier_index <- find_min_features_classifier(classifiers)
    classifier <- classifiers[[min_features_classifier_index]]
    
    # If this is not the first iteration, update target variable
    if (!is.null(classifier)) {
      binary_target <- ifelse(target == classifier$class_label, classifier$class_label, "OTHERS")
      binary_target <- as.factor(binary_target)
    }
    
    # Train or retrain the classifier
    if (is.null(classifier)) {
      # First iteration: Train the classifier for the class with the least features
      classifier <- train_classifier(data, target, "ALL")
      classifiers[[length(classifiers) + 1]] <- classifier
    } else {
      # Subsequent iterations: Retrain the classifier for the class with the least features
      classifier <- train_classifier(data, target, classifier$class_label)
      classifiers[[min_features_classifier_index]] <- classifier
    }
    
    # Predict using the current classifier
    predictions <- predict_classifier(classifier$model, data)
    
    # Update predicted labels based on the current classifier
    predicted_labels[is.na(predicted_labels) & predictions != "OTHERS"] <- classifier$class_label
  }
  
  return(predicted_labels)
}

# Train classifiers for each class against OTHERS
for (class_label in levels(target)) {
  if (class_label != "OTHERS") {
    classifier <- train_classifier(data, target, class_label)
    classifier$class_label <- class_label
    classifiers[[length(classifiers) + 1]] <- classifier
  }
}

# Perform sequential classification
predicted_labels <- sequential_classification(data, target, classifiers)

# Evaluate overall accuracy
overall_accuracy <- evaluate_classifier(predicted_labels, target)
print(paste("Overall Accuracy:", overall_accuracy))
