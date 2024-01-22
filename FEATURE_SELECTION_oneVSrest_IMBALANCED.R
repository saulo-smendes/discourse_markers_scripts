# ANEXO D1.3 Leaps and bounds e LDA
library("readxl")
library("MASS")
library("caret")
library("cluster")
#library("pgirmess")
install.packages("caret")
install.packages("readxl", "MASS")
library(rpart)
library(rpart.plot)
library(caret)
library(e1071)
library(yardstick)

set.seed(123)

# Install and load the leaps package
if (!requireNamespace("leaps", quietly = TRUE)) {
  install.packages("leaps")
}
library("leaps")


# Create a list containing the names of the classes
classes <- c("ALL", "CNT", "EVD", "EXP", "INP")


# Iterate over the classes and perform feature selection
for (target_dm in classes) {
  
  # Print current class
  print(target_dm)
  
  title <- paste(target_dm, "vs OTHER", sep = " ")
  
  # Load the data from a csv file in "C:\Users\Saulo Mendes Santos\OneDrive\Documents\2. LETRAS\0. Doutorado\0. Recherche\1.5. Classification"
  #data <- read.csv("C:/Users/Saulo Mendes Santos/OneDrive/Documents/2. LETRAS/0. Doutorado/0. Recherche/1.5. Classification/results_dm_vfinal_standardized_withf0coeff_20231205.csv")
  data <- read.csv('/Volumes/LaCie/DOUTORADO/1.5. Classification/results_dm_vfinal_standardized_withf0coeff_20231205.csv')
  
  # We remap the target variable so that we only have two classes: 'ALL' and 'OTHER'
  data$iu <- ifelse(data$iu == target_dm, target_dm, "OTHER")
  
  
  # Get the number of rows and columns in data
  dim(data)
  
  # Extract features and target variable
  # Features are contained in column 6 to 40
  features <- data[, 11:40]
  
  # Target in iu
  target <- data$iu
  
  # Encode my target variable as a factor
  target <- as.factor(target)
  
  # Get names of selected features
  feature_names <- names(features)
  
  # Check types of features
  sapply(features, class)
  
  # Imputation of missing values in features
  # Replace missing values with the mean of the column
  features <- apply(features, 2, function(x) replace(x, is.na(x), mean(x, na.rm = TRUE)))

  
  
  #### OLIVER'S SCRIPT ####
  REGS <- regsubsets(features, target, int = TRUE, names = feature_names, nbest = 60, nvmax = 30, really.big = TRUE)
  
  RegsW <- summary(REGS)$which
  RegsS <- as.numeric(row.names(RegsW))
  
  # Save plot
  name_cp <- paste("cp_stata_", target_dm)
  name_cp <- paste(name_cp, ".png")
  png(file = name_cp)
  plot(RegsS, summary(REGS)$cp, col = "#008810", log = "y",
       xlab = "Number of parameters", ylab = "Cp Stat", main=title, cex.lab = 1.2)
  points(1:nrow(features), 1:nrow(features), type = "l", col = 6, cex = 0.7)
  legend(x = nrow(features) - 7, y = 0.3, legend = c("Cp = p"), col = 6, lty = 1, bty ="n")
  dev.off()
  
  
  name_plot <- paste("n_selection_", target_dm)
  name_plot <- paste(name_plot, ".png")
  png(file=name_plot, width=700)
  par(las = 2)
  par(mar = c(15, 6, 2, 1))
  barplot(sort(colSums(RegsW)/nrow(RegsW)*100, decreasing = TRUE)[-1], col = 3, ylim = c(0, 100),
          ylab = "Times feature was selected", cex.lab = 1.2, main=title)
  dev.off()
  #Save plot
  
  
  nmod <-nrow(RegsW)
  TACC <- rep(list(matrix(0, nmod, length(levels(target)))), 11)
  OACC <- rep(0, nmod)
  
  # We will build a dataframe called "lUDdat" with features and target variable. The target variable will be placed in a column called "Tag"
  lUDdat <- data.frame("Tag" = target, features)
  
  # Run LDA
  for (n in 1:nmod){
    LDA <- lda(Tag ~., data = lUDdat[, c("Tag", names(which(RegsW[n,]))[-1])])
    mTag <- predict(LDA, lUDdat)$class
    conf <- confusionMatrix(mTag, lUDdat$Tag, mode="everything")
    OACC[n] <- conf$byClass[7]
    #OACC[n] <- conf$overall[1]
    
    #for (m in 1:11) TACC[[m]][n,] <- conf$byClass[, m]
  }
  par(xaxs="i")
  lsVAR <- names(which(RegsW[which.max(OACC),]))[-1]
  LDA <- lda(Tag ~., data = lUDdat[, c("Tag", lsVAR)])
  mTag <- predict(LDA, lUDdat)$class
  conf <- confusionMatrix(mTag, lUDdat$Tag, mode="everything")
  
  # Print lsVAR and conf
  print(lsVAR)
  print(conf)
  # Print max f1 score of the LDA model
  print("Max f1 score of the LDA model:")
  print(conf$byClass[7])
  
  
  
  #### For the decision tree
  # Extract the relevant data
  data_subset <- lUDdat[, c("Tag", lsVAR)]
  
  # Build the Decision Tree model using all data
  tree_model <- rpart(Tag ~ ., data = data_subset, method = "class")
  
  # Make predictions on the same data
  predictions <- predict(tree_model, data_subset, type = "class")
  
  # Evaluate the model
  conf_matrix <- confusionMatrix(predictions, data_subset$Tag, mode = "everything")
  print(conf_matrix)
  
  # Plot the Decision Tree
  name_dt <- paste("dt_plot_", target_dm)
  name_dt <- paste(name_dt, ".png")
  png(file = name_dt, width = 800, height = 600, units = "px", res = 300)
  rpart.plot(tree_model, box.palette = "RdBu", shadow.col = "gray", nn = TRUE)
  dev.off()
  
  # Print the gini index of the Decision Tree
  print("Gini index of the Decision Tree:")
  tree_gini <- 1 - sum((tree_model$frame$dev) ^ 2)
  print(tree_gini)
  
  # Print the f1 score of the Decision Tree
  print("f1 score of the Decision Tree:")
  tree_f1 <- conf_matrix$byClass[7]
  print(tree_f1)
  
  
  # Save results of lsVAR and conf
  #write.csv(lsVAR, paste("C:/Users/Saulo Mendes Santos/OneDrive/Documents/2. LETRAS/0. Doutorado/0. Recherche/1.5. Classification/feature_selection/feature_selection_oneVSrest_", target_dm, "_lsVAR.csv", sep = ""))
  
  #write.csv(conf, paste("C:/Users/Saulo Mendes Santos/OneDrive/Documents/2. LETRAS/0. Doutorado/0. Recherche/1.5. Classification/feature_selection/feature_selection_oneVSrest_", target_dm, "_conf.csv", sep = ""))
  
}
