data <- read.csv(file.choose())
data

    #1 BORUTA

# install.packages('Boruta')
library(Boruta)

# Perform Boruta search
boruta_output <- Boruta(#targetvariable ~ ., data=na.omit(data), doTrace=0)

# Get significant variables including tentatives
boruta_signif <- getSelectedAttributes(boruta_output, withTentative = TRUE)
print(boruta_signif)  

# Do a tentative rough fix
roughFixMod <- TentativeRoughFix(boruta_output)
boruta_signif <- getSelectedAttributes(roughFixMod)
print(boruta_signif)

# Variable Importance Scores
imps <- attStats(roughFixMod)
imps2 = imps[imps$decision != 'Rejected', c('meanImp', 'decision')]
head(imps2[order(-imps2$meanImp), ])  # descending sort

# Plot variable importance
plot(boruta_output, cex.axis=.7, las=2, xlab="", main="Variable Importance")  

--------

    #2 Variable Importance from Machine Learning Algorithms

# Train an rpart model and compute variable importance.
library(caret)
set.seed(100)
colnames(data) <- make.names(colnames(data))
rPartMod <- train(#targetvariable ~ ., data=data, method="rpart")
rpartImp <- varImp(rPartMod)
print(rpartImp)

# Train an RRF model and compute variable importance.
set.seed(100)
rrfMod <- train(Class ~ ., data=trainData, method="RRF")
rrfImp <- varImp(rrfMod, scale=F)
rrfImp
plot(rrfImp, top = 20, main='Variable Importance')

--------

    #3 Lasso Regression
    
library(glmnet)
trainData <- data

x <- as.matrix(trainData[,-63]) # all X vars
y <- as.double(as.matrix(ifelse(trainData[, 63]=='normal', 0, 1))) # Only Class

# Fit the LASSO model (Lasso: Alpha = 1)
set.seed(100)
cv.lasso <- cv.glmnet(x, y, family='binomial', alpha=1, parallel=TRUE, standardize=TRUE, type.measure='auc')

# Results
plot(cv.lasso)

# plot(cv.lasso$glmnet.fit, xvar="lambda", label=TRUE)
cat('Min Lambda: ', cv.lasso$lambda.min, '\n 1Sd Lambda: ', cv.lasso$lambda.1se)
df_coef <- round(as.matrix(coef(cv.lasso, s=cv.lasso$lambda.min)), 2)

# See all contributing variables
df_coef[df_coef[, 1] != 0, ]

--------

    #4 Step wise Forward and Backward Selection
    
 # Load data
trainData <- data
print(head(trainData))
# Step 1: Define base intercept only model
base.mod <- lm(ozone_reading ~ 1 , data=trainData)  

# Step 2: Full model with all predictors
all.mod <- lm(ozone_reading ~ . , data= trainData) 

# Step 3: Perform step-wise algorithm. direction='both' implies both forward and backward stepwise
stepMod <- step(base.mod, scope = list(lower = base.mod, upper = all.mod), direction = "both", trace = 0, steps = 1000)  

# Step 4: Get the shortlisted variable.
shortlistedVars <- names(unlist(stepMod[[1]])) 
shortlistedVars <- shortlistedVars[!shortlistedVars %in% "(Intercept)"] # remove intercept

# Show
print(shortlistedVars)

--------

    #5 Relative Importance from Linear Regression
 
# install.packages('relaimpo')
library(relaimpo)

# Build linear regression model
model_formula = ozone_reading ~ Temperature_Sandburg + Humidity + Temperature_ElMonte + Month + pressure_height + Inversion_base_height
lmMod <- lm(model_formula, data=trainData)

# calculate relative importance
relImportance <- calc.relimp(lmMod, type = "lmg", rela = F)  

# Sort
cat('Relative Importances: \n')
sort(round(relImportance$lmg, 3), decreasing=TRUE)
bootsub <- boot.relimp(ozone_reading ~ Temperature_Sandburg + Humidity + Temperature_ElMonte + Month + pressure_height + Inversion_base_height, data=trainData,
                       b = 1000, type = 'lmg', rank = TRUE, diff = TRUE)

plot(booteval.relimp(bootsub, level=.95))

--------

    #6 Recursive Feature Elimination (RFE)
    
str(trainData)
set.seed(100)
options(warn=-1)

subsets <- c(1:5, 10, 15, 18)

ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)

lmProfile <- rfe(x=trainData[, c(1:3, 5:13)], y=trainData$ozone_reading,
                 sizes = subsets,
                 rfeControl = ctrl)

lmProfile

--------

    #7 Genetic Algorithm

# Define control function
ga_ctrl <- gafsControl(functions = rfGA,  # another option is `caretGA`.
                        method = "cv",
                        repeats = 3)

# Genetic Algorithm feature selection
set.seed(100)
ga_obj <- gafs(x=trainData[, c(1:3, 5:13)], 
               y=trainData[, 4], 
               iters = 3,   # normally much higher (100+)
               gafsControl = ga_ctrl)

ga_obj

# Optimal variables
ga_obj$optVariables

--------

    #8 Simulated Annealing
    
# Define control function
sa_ctrl <- safsControl(functions = rfSA,
                        method = "repeatedcv",
                        repeats = 3,
                        improve = 5) # n iterations without improvement before a reset

# Genetic Algorithm feature selection
set.seed(100)
sa_obj <- safs(x=trainData[, c(1:3, 5:13)], 
               y=trainData[, 4],
               safsControl = sa_ctrl)

sa_obj

# Optimal variables
print(sa_obj$optVariables)

--------

    #9 Information Value and Weights of Evidence
    
 library(InformationValue)
inputData <- read.csv(" ")
print(head(inputData))
# Choose Categorical Variables to compute Info Value.
cat_vars <- c ("WORKCLASS", "EDUCATION", "MARITALSTATUS", "OCCUPATION", "RELATIONSHIP", "RACE", "SEX", "NATIVECOUNTRY")  # get all categorical variables

# Init Output
df_iv <- data.frame(VARS=cat_vars, IV=numeric(length(cat_vars)), STRENGTH=character(length(cat_vars)), stringsAsFactors = F)  # init output dataframe

# Get Information Value for each variable
for (factor_var in factor_vars){
  df_iv[df_iv$VARS == factor_var, "IV"] <- InformationValue::IV(X=inputData[, factor_var], Y=inputData$ABOVE50K)
  df_iv[df_iv$VARS == factor_var, "STRENGTH"] <- attr(InformationValue::IV(X=inputData[, factor_var], Y=inputData$ABOVE50K), "howgood")
}

# Sort
df_iv <- df_iv[order(-df_iv$IV), ]

df_iv

WOETable(X=inputData[, 'WORKCLASS'], Y=inputData$ABOVE50K)

--------

    #10 DALEX package 
    
library(randomForest)
library(DALEX)

# Load data
inputData <- read.csv("http://rstatistics.net/wp-content/uploads/2015/09/adult.csv")

# Train random forest model
rf_mod <- randomForest(factor(ABOVE50K) ~ ., data=inputData, ntree=100)
rf_mod

# Variable importance with DALEX
explained_rf <- explain(rf_mod, data=inputData, y=inputData$ABOVE50K)

# Get the variable importances
varimps = variable_dropout(explained_rf, type='raw')

print(varimps)

plot(varimps)




