# Intro -------------------------------------------------------------------

# Load Libraries
suppressWarnings(
  suppressMessages(
    {
      library(tidyverse, quietly = T)
      library(rBayesianOptimization, quietly = T)
      library(xgboost, quietly = T)
    }
  )
)

# Load Data
# Data comes after Feature Engineering and Missing Imputation
load('ames_frame.rda')

summary(ames)
cat(
  'Count of Variables with No Missing Values = ',
  table(apply(is.na(ames),2,sum) == 0),
  '\n'
  )


# Model Development -------------------------------------------------------

train = ames %>% subset(.,SalePrice > 0) %>% select(-Date)
y = log(train$SalePrice)
dtrain <- xgb.DMatrix(model.matrix(SalePrice ~ ., train),label = y)
cv_folds <- KFold(y, nfolds = 5, stratified = F, seed = 2018)

xgb_cv_bayes <- function(max.depth, min_child_weight, subsample) {
  cv <- xgb.cv(
    params = list(
      booster = "gbtree", 
      eta = 0.15,
      max_depth = max.depth,
      min_child_weight = min_child_weight,
      subsample = subsample,
      colsample_bytree = 0.6,
      lambda = .01,
      alpha = 0,
      gamma = 0.03,
      objective = "reg:linear",
      eval_metrix = "rmse"
      ),
    data = dtrain, 
    nround = 100,
    folds = cv_folds, 
    prediction = FALSE, 
    showsd = FALSE,
    early_stop_round = 5,
    maximize = FALSE,
    verbose = 0
    )
  return(list(Score = -min(cv$evaluation_log$test_rmse_mean)))
} 

OPT_Res = BayesianOptimization(
  xgb_cv_bayes,
  bounds = list(
    max.depth = c(2L, 10L),
    min_child_weight = c(1L, 20L),
    subsample = c(0.5, 0.8)
    ),
  init_grid_dt = NULL, 
  init_points = 10, 
  n_iter = 50,
  acq = "ei", 
  kappa = 2.576, 
  eps = 0.1,
  verbose = TRUE
  )