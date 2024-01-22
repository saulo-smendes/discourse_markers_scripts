# ANEXO D1.3 Leaps and bounds e LDA
library("readxl")
library("MASS")
library("caret")
library("cluster")
#library("pgirmess")
library(rpart)
library(rpart.plot)
library(e1071)
library(yardstick)

# Install and load the leaps package
if (!requireNamespace("leaps", quietly = TRUE)) {
  install.packages("leaps")
}
library("leaps")


# Load the data from a csv file in "C:\Users\Saulo Mendes Santos\OneDrive\Documents\2. LETRAS\0. Doutorado\0. Recherche\1.5. Classification"
#data <- read.csv("C:/Users/Saulo Mendes Santos/OneDrive/Documents/2. LETRAS/0. Doutorado/0. Recherche/1.5. Classification/results_dm_vfinal_standardized_withf0coeff_20231205.csv")
data <- read.csv('/Volumes/LaCie/DOUTORADO/1.5. Classification/results_dm_vfinal_standardized_withf0coeff_20231205.csv')


# Extract features and target variable
# Features are contained in column 6 to 40
features <- data[, 11:40]

# Exclude columns from 2 to 4
#features <- features[, -c(2:5)]

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
#var_d <-var[var$Var%in%names(dUDdat),]
#var_d <- features
#var_l <- var_d[which(var_d$t2!="s" & var_d$t2!="i" & var_d$t2 != "x" & var_d$t1 != "i"), ]


#REGS <- regsubsets(lUDdat[, -1], lUDdat$Tag, int = TRUE, names = names(lUDdat[-1]), nbest = 60, nvmax = 30, really.big = TRUE)

REGS <- regsubsets(features, target, int = TRUE, names = feature_names, nbest = 60, nvmax = 30, really.big = TRUE)

RegsW <- summary(REGS)$which
RegsS <- as.numeric(row.names(RegsW))
plot(RegsS, summary(REGS)$cp, col = "#008810", log = "y",
     xlab = "Number of parameters", ylab = "Cp Stat", cex.lab = 1.2)
points(1:nrow(features), 1:nrow(features), type = "l", col = 6, cex = 0.7)
legend(x = nrow(features) - 7, y = 0.3, legend = c("Cp = p"), col = 6, lty = 1, bty ="n")
par(las = 2)
par(mar = c(15, 6, 2, 1))
barplot(sort(colSums(RegsW)/nrow(RegsW)*100, decreasing = TRUE)[-1], col = 3, ylim = c(0, 100),
        ylab = "Number of times the variable was select (%)", cex.lab = 1.2)
nmod <-nrow(RegsW)
TACC <- rep(list(matrix(0, nmod, length(levels(target)))), 11)
OACC <- rep(0, nmod)

# We will build a dataframe called "lUDdat" with features and target variable. The target variable will be placed in a column called "Tag"
lUDdat <- data.frame("Tag" = target, features)

# Run LDA
for (n in 1:nmod){
  LDA <- lda(Tag ~., data = lUDdat[, c("Tag", names(which(RegsW[n,]))[-1])])
  mTag <- predict(LDA, lUDdat)$class
  conf <- confusionMatrix(mTag, lUDdat$Tag, mode = "everything")
  OACC[n] <- conf$overall[1]
  f1scores <- conf$byClass[, 7]
  
  for (m in 1:11) TACC[[m]][n,] <- conf$byClass[, m]
}
par(xaxs="i")
lsVAR <- names(which(RegsW[which.max(OACC),]))[-1]
LDA <- lda(Tag ~., data = lUDdat[, c("Tag", lsVAR)])
print(LDA)
plot(LDA)
mTag <- predict(LDA, lUDdat)$class
conf <- confusionMatrix(mTag, lUDdat$Tag, mode = "everything")
lsVAR; conf

# REscale the data and run LDA again
sdata <- data.frame("Tag" = lUDdat$Tag, scale(lUDdat[, lsVAR]))
sLDA <- lda(Tag ~., data = sdata)
print(sLDA)



#### DECISION TREE MODEL WITH 5 CLASSES ####
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
name_dt <- "dt_plot_all-features_imbalanced.png"
png(file = name_dt, width = 800, height = 600, units = "px", res = 300)
rpart.plot(tree_model, box.palette = "RdBu", shadow.col = "gray", nn = TRUE)
dev.off()



#### OTHER PLOTS FOR THE LDA MODEL ####
par(mar = c(4, 5.3, 2, 1))
par(mgp = c(3, 0.5,0))

cores = rgb(t((col2rgb(as.numeric(lUDdat$Tag) + 1) + col2rgb(as.numeric(mTag) + 1))/2/255))
plot(LDA, col = cores, main = paste(c(length(lsVAR), "p:", lsVAR), collapse = " "),
     ylim = c(-3, 3), xlim = c(-4, 4), cex.lab = 3, cex = 2, cex.main = 3, cex.axis = 3)
pot <- LDA$svd^2/sum(LDA$svd^2)
sdata <- data.frame("Tag" = lUDdat$Tag, scale(lUDdat[, lsVAR]));
sLDA <- lda(Tag ~., data = sdata)
cLDA <- cbind(coef(sLDA), coef(LDA))
cLDA <- cLDA[order(cLDA[, 1], decreasing = TRUE), ]
write.table(cLDA, "clipboard", sep = "\t")
nv <- REGS$np - 1

# This must be for each class and global
# Global accuracy
maxoa <- sapply(1:nv, function(x) max(OACC[which(RegsS == x)]))
# Accuracy for each class
# ALL
maxaa <- sapply(1:nv, function(x) max(TACC[[11]][which(RegsS == x), 1]))
# CNT
maxca <- sapply(1:nv, function(x) max(TACC[[11]][which(RegsS == x), 2]))

# EXP
maxea <- sapply(1:nv, function(x) max(TACC[[11]][which(RegsS == x), 3]))

# EVD
maxva <- sapply(1:nv, function(x) max(TACC[[11]][which(RegsS == x), 4]))

# INP
maxia <- sapply(1:nv, function(x) max(TACC[[11]][which(RegsS == x), 5]))


# Plot the results of accuracy by the number of parameters
par(mgp = c(2, 0.5, 0))
par(mar = c(3, 3.5, 2, 1))
plot(1:nv, maxoa, col = 1, type = "b", ylim = c(0.5, 1), xlab = "Number of parameters",
     ylab = "Max accuracy score (%)", cex.lab = 1.2, frame.plot = FALSE, xlim = c(0, 27))

# Plot lines with thicker lines
lines(1:nv, maxaa, col = 2, type = "b", lwd = 2)
lines(1:nv, maxca, col = 3, type = "b", lwd = 2)
lines(1:nv, maxea, col = 4, type = "b", lwd = 2)
lines(1:nv, maxva, col = 5, type = "b", lwd = 2)
lines(1:nv, maxia, col = 6, type = "b", lwd = 2)

# Customize the legend with thicker lines
legend(x = nv - 8, y = 0.7, legend = c("Global", "ALL", "CNT", "EXP", "EVD", "INP"), 
       col = 1:6, lty = 1, lwd = 2, bty = "n")

# Add gridlines
grid(nx = NA, ny = NULL, col = "lightgray", lty = "dotted", lwd = par("lwd"), equilogs = TRUE)




k = 10
kRegsW <- RegsW[which(RegsS == k),]
kOACC <- OACC[which(RegsS == k)]
kVAR <- names(which(kRegsW[which.max(kOACC),]))[-1]
kLDA <- lda(Tag ~., data = lUDdat[, c("Tag", kVAR)])
kVAR
kPRED <- predict(kLDA, lUDdat)
kconf <- confusionMatrix(kPRED$class, lUDdat$Tag)
kconf
cores <- rgb(t((col2rgb(as.numeric(lUDdat$Tag) + 1) + col2rgb(as.numeric(kPRED$class) + 1))/2/255))
plot(kLDA, col = cores, main = paste(c(k, "p", ":", kVAR), collapse = " "))
if (sum(max(kOACC)==kOACC) > 1) {
  t(sapply(1: nrow(kRegsW[which(kOACC == max(kOACC)),]), function(x) names(which(kRegsW[which(kOACC == max(kOACC)),][x,]))[-1]))
}



#### reviewed plot discriminants ####
# Assuming you have defined your variables and performed the necessary calculations above...

k = 20
kRegsW <- RegsW[which(RegsS == k),]
kOACC <- OACC[which(RegsS == k)]
kVAR <- names(which(kRegsW[which.max(kOACC),]))[-1]
kLDA <- lda(Tag ~., data = lUDdat[, c("Tag", kVAR)])
kVAR
kPRED <- predict(kLDA, lUDdat)
kconf <- confusionMatrix(kPRED$class, lUDdat$Tag)
kconf
cores <- rgb(t((col2rgb(as.numeric(lUDdat$Tag) + 1) + col2rgb(as.numeric(kPRED$class) + 1))/2/255))

# Add a title to the plot
plot(kLDA, col = cores, collapse = " ")

# Add a title to the plot
title(main = paste("LDA Plot for ", k, "features"), line = 1, cex.main = 1.2)

# Check if max OACC has ties
if (sum(max(kOACC) == kOACC) > 1) {
  t(sapply(1: nrow(kRegsW[which(kOACC == max(kOACC)),]), function(x) names(which(kRegsW[which(kOACC == max(kOACC)),][x,]))[-1]))
}

