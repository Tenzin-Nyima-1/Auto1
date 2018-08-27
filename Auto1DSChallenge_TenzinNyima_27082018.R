# Date: 27 August 2018;
# Data Science challenge: Auto1;
# Submitted by: Tenzin Nyima;
# Goal: to build a statistical model that best predicts the price of cars-; 
#-that's in the best interest of Auto1's business;
# Using: R version 3.5.0 (2018-04-23);
# Workflow;
# I.Data Handling;
# II.Exploratory Analysis;
# III. Fitting Models;
# IV. Model Comparison;
# V. Variable Importance;
# VI. Save Worksape;


# I. Packages ____________________________________________________________;
library(caret)
library(psych)
library(corrplot)


# II. Data Handling ___________________________________________________;
setwd("~/Documents/Auto1_DataChallenge/Data")

# 1. load the data;
Auto1Data <- read.csv("Auto1-DS-TestData.csv", stringsAsFactors = FALSE)
dim(Auto1Data)
# [1] 205  26;

# 1a. structure;
summary(Auto1Data)
str(Auto1Data)

# 2. identify NAs;
Auto1Data[Auto1Data == "?"] <- NA
Auto1Data[Auto1Data == ""] <- NA

# 2a. omitting NAs;
# Reason: as a first run of the analysis with a full data;
Auto1Data <- na.omit(Auto1Data)
dim(Auto1Data)
# [1] 159  26;

# 3. Removing uninteresting/NA cols;
NACols  <- c("engine.location","fuel.system","engine.type")
# @Reasons: 1st - only one level; 2-3: error "rank-deficient fit";

# 3a. clean S1;
Auto1Data_1 <- Auto1Data[, -(match(NACols, colnames(Auto1Data)))]
dim(Auto1Data_1)
# [1] 159  23;

# str(Auto1Data_1)
# summary(Auto1Data_1)

# 3b. Convert numeric cols to numeric;
numCols <- c("symboling", "normalized.losses", "wheel.base", "length", "width", "height", "curb.weight", "engine.size", "bore","stroke", "compression.ratio", "horsepower", "peak.rpm", "city.mpg", "highway.mpg", "price")

for(i in seq(numCols)){
  Auto1Data_1[,numCols[i]] <- as.numeric(Auto1Data_1[,numCols[i]])
  }
# str(Auto1Data_1)
# summary(Auto1Data_1)

# 3c. Convert char cols to char;
charCols <- setdiff(colnames(Auto1Data_1), numCols)

# @crosscheck;
identical(ncol(Auto1Data_1), length(unique(c(numCols, charCols))))
# [1] TRUE;

for(i in seq(charCols)){
  Auto1Data_1[,charCols[i]] <- as.character(Auto1Data_1[,charCols[i]])
}
# str(Auto1Data_1)
# summary(Auto1Data_1)

# 4. remove values with only one level;
# Note: to avoid errors due to such values;
charsObsCounts <- vector(mode = "list", length = length(charCols))
names(charsObsCounts) <- charCols

# 4a. identify those obs;
#S1: count values per level in each variable;
for (i in seq(charsObsCounts)) {
  charsObsCounts[[i]] <- table(factor(Auto1Data_1[,names(charsObsCounts)[i]], 
                                      levels = unique(Auto1Data_1[,names(charsObsCounts)[i]])))
  }
#S2: in each variable, get index of levels with <=6 values;
charsObsCounts_1 <- lapply(charsObsCounts, function(x) which(x<=6))
#S2: variable names containing levels with <=6 values;
charsObsCounts_2 <- charsObsCounts_1[unlist(lapply(charsObsCounts_1, function(x) length(x) != 0))]
#S3: get indexes of values with <=6 values in each level;
charsObsCounts_2Ind <- vector("list", length = length(charsObsCounts_2))
for (i in seq(length(charsObsCounts_2))) {
  charsObsCounts_2Ind[[i]]<-which(!is.na(match(Auto1Data_1[,names(charsObsCounts_2)[i]], names(charsObsCounts_2[[i]]))))
}
#S4: final data, after removing rows with <=6 values in each level;
Auto1DataFin <- Auto1Data_1[-(unique(unlist(charsObsCounts_2Ind))), ]
dim(Auto1DataFin)
# [1] 124  23;
#S5: convert char cols into factors;
for (i in seq(charCols)) {
  Auto1DataFin[,charCols[i]] <- factor(Auto1DataFin[,charCols[i]], levels = unique(Auto1DataFin[,charCols[i]]))
}
#summary(Auto1DataFin)
#str(Auto1DataFin)


# 5. Save data separately;

# @Y(targets);
Auto1DataFin_Y <- Auto1DataFin$price
length(Auto1DataFin_Y)
# [1] 124;

# @X(predictors);
Auto1DataFin_X <- Auto1DataFin[,-match("price", colnames(Auto1DataFin))]
dim(Auto1DataFin_X)
# [1] 124  22;


# III. Exploratory analysis ___________________________________________;

# 1. summary statistics of raw data after omitting NAs; 
summary(Auto1Data)
str(Auto1Data)

# 2. summary statistics of organized data after omitting NAs; 
summary(Auto1DataFin)
str(Auto1DataFin)

# 3. check the distribution of raw, cleaned and log transformed prices;
par(mfrow=c(1,3))
hist(as.numeric(Auto1Data$price))
hist(Auto1DataFin$price)
hist(log(Auto1DataFin$price))
dev.off()

# 4. correlation between price and the predictors;
Auto1DataFin_Cor <- corr.test(Auto1DataFin[,names(which(sapply(Auto1DataFin, function(x) is.numeric(x))))],
                              method="pearson", adjust="BH")

# @visualize;
corrplot(Auto1DataFin_Cor$r, method = "number", type = "upper", number.cex = .5)


# III. Fitting models___________________________________________________;
# @Y(targets);
length(Auto1DataFin_Y)
# [1] 124;
# @X(predictors);
dim(Auto1DataFin_X)
# [1] 124  22;

# 1. To fit models with similar training and test splits, set up the train control object;
Auto1DataFin_TrCont <- trainControl(method = "cv", number = 10, verboseIter = TRUE)

# 2. Generalized Linear Model;

# 2.1 without preprocessing;
set.seed(72)
Auto1DataFin_GLM <- train(x = Auto1DataFin_X, y = Auto1DataFin_Y, 
                          method = "glm", 
                          trControl = Auto1DataFin_TrCont)
Auto1DataFin_GLM$results$RMSE
# [1] 1542.653;

# 2.2 preprocessing 1;
set.seed(72)
Auto1DataFin_GLM_pp1 <- train(x = Auto1DataFin_X, y = Auto1DataFin_Y, 
                          method = "glm", 
                          trControl = Auto1DataFin_TrCont,
                          preProcess = c("zv", "center", "scale", "pca"))
Auto1DataFin_GLM_pp1$results$RMSE
# [1] 1490.604;

# 2.3 preprocessing 2;
set.seed(72)
Auto1DataFin_GLM_pp2 <- train(x = Auto1DataFin_X, y = Auto1DataFin_Y, 
                              method = "glm", 
                              trControl = Auto1DataFin_TrCont,
                              preProcess = c("zv", "center", "scale", "spatialSign"))
Auto1DataFin_GLM_pp2$results$RMSE
# [1] 1492.925;

# 3. Random Forest;
RF_Mtry <- data.frame(mtry=1:15)
set.seed(72)
Auto1DataFin_RF <- train(x = Auto1DataFin_X, y = Auto1DataFin_Y, 
                         tuneGrid = RF_Mtry, method = "rf", 
                         trControl = Auto1DataFin_TrCont)
plot(Auto1DataFin_RF)


# IV. Model comparison___________________________________________________;

# 1. Model lists;
Auto1DataFin_ModLS <- list(GM = Auto1DataFin_GLM_pp1, RF = Auto1DataFin_RF)

# 2. Resample ls;
Auto1DataFin_ModLS_reSmp <- resamples(Auto1DataFin_ModLS)
summary(Auto1DataFin_ModLS_reSmp)

# 3. display;
bwplot(Auto1DataFin_ModLS_reSmp, metric = "RMSE", main = "Model comparison")

# 4. NOTE;
# GLM model with preprocess is slightly better than the RF model, therefore the former is chosen;


# V. Variable importance in preprocessed GLM model ________________________;

# 1. Detailed summary of the selected model;
summary(Auto1DataFin_GLM_pp1)

# 2. Display Var imp;
Auto1DataFin_GLM_pp1_VImp <- varImp(Auto1DataFin_GLM_pp1, scale=TRUE)
plot(Auto1DataFin_GLM_pp1_VImp, main = "Variable Importance in GLM model, preprocessed(1)")


# VI. Save workspace ______________________________________________________;
# save.image("~/Documents/Auto1_DataChallenge/RScript/Auto1DSChallenge_TenzinNyima_27082018.RData")
# load("~/Documents/Auto1_DataChallenge/RScript/Auto1DSChallenge_TenzinNyima_27082018.RData")
# End ######################################################################;
