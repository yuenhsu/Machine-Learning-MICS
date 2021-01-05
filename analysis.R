##===============================================================
##                            Setup                            ==
##===============================================================

##---------------------------
##  Load Libraries and Data  
##---------------------------
library(tidyverse)
library(haven)
library(caret)
library(rpart.plot)
library(arsenal)
library(gam)
library(randomForest)
mics <- read_dta("MICS.dta")
summary(mics)

##--------------------------
##  Order & Rename Columns  
##--------------------------
mics <- mics %>% rename(
  "age" = "AGE",
  "edu" = "HIGHEST_EDU",
  "nchild" = "CHILDREN_BORN",
  "mstat" = "MARITAL_STATUS",
  "residence" = "URBAN",
  "country" = "COUNTRY",
  "given_birth" = "GIVEN_BIRTH",
  "child_died" = "CHILD_DIED",
  "use" = "CONTRACT",
  "wealth" = "WEALTH_PERCT"
)
str(mics, give.attr = F)
##----------------------
##  Organise Variables  
##----------------------
# Extract number from wealth (percentile) and convert to numerical variable
mics$wealth <- as.numeric(str_extract(mics$wealth,"\\d+"))

# Factor variables
mics$mstat <- str_replace(mics$mstat, "ly married/in union","")
mics$mstat <- str_replace(mics$mstat, " married/in union","")
mics$mstat <- factor(mics$mstat)
mics$edu <- factor(mics$edu)
mics$country <- factor(mics$country)

# Factor recode for clarity
mics$use <- factor(mics$use, 
                   levels = c(1, 0), 
                   labels = c("Using", "Not Using"))

mics$residence <- factor(mics$residence, 
                         levels = c(1, 0), 
                         labels = c("Urban", "Rural"))

mics$given_birth <- factor(mics$given_birth, 
                           levels = c(1, 0),
                           labels = c("Yes", "No"))

mics$child_died <- factor(mics$child_died, 
                          levels = c(1, 0),
                          labels = c("Yes", "No"))

mics$edu <- factor(mics$edu, 
                   levels = c("PRE-PRIMARY OR NONE", "PRIMARY",
                              "LOWER SECONDARY", "UPPER SECONDARY", "HIGHER"),
                   labels = c("Less than Primary", "Primary",
                              "Lower Secondary", "Upper Secondary", 
                              "Higher Education"))

mics$country <- factor(mics$country,
                       levels = c("THAILAND", "LAO", "MONGOLIA"),
                       labels = c("Thailand", "Laos", "Mongolia"))
str(mics, give.attr = F)
summary(tableby(use~., data = mics,
                control = tableby.control(numeric.stats = c("mean","iqr","median"),
                                          cat.stats = c("countrowpct"),
                                          test = FALSE)))

##================================================================
##                     Predictive Modelling                     ==
##================================================================

##------------------------
##  Split Validation Set  
##------------------------
set.seed(20)
# Split data into testing and training
train_index <- createDataPartition(mics$use,
                                   # 85% for training
                                   p = .85,
                                   times = 1,
                                   list = FALSE)

micsTrain <- mics[ train_index, ]
micsTest  <- mics[-train_index, ]

# Check outcome distribution
prop.table(table(mics$use))
prop.table(table(micsTrain$use))


##----------------------------
##  10-fold Cross-Validation  
##----------------------------
set.seed(20)
# Index for each fold
fold_index <- createFolds(micsTrain$use,
                          # number of folds
                          k = 10, 
                          # return as list
                          list = T, 
                          # return numbers corresponding to the positions
                          returnTrain = T)

# Cross validation command
ctrl <- trainControl(method="cv", index = fold_index)

# Store result
cv_result <- tibble(
  Model = character(),
  Param = integer(),
  cv = numeric()
)



##--------------------
##  1. KNN  
##  Approx. 7-10 min
##--------------------
set.seed(20)
modelLookup("knn")
start.time <- Sys.time()

m_knn <- train(form = use~.,
               data = micsTrain,
               method = 'knn',
               trControl = ctrl, # Cross-validation
               tuneLength = 10)

end.time <- Sys.time()
print(end.time - start.time)

m_knn # k = 9, cv = .7423
#write.csv(tibble(m_knn$results), "20201122_mod2_knn.csv")
plot(m_knn, main = "KNN 10-fold Cross-Validation")


cv_result <-
  cv_result %>%
  add_row(tibble(Model = 'KNN',
                 Param = 9,
                 cv = .7423))


##--------------------
## 2. Decision Tree
## Approx. 10-30 sec
##--------------------
set.seed(20)

start.time <- Sys.time()

m_tree <- train(
  use ~. -nchild,
  micsTrain,
  method = "rpart2",
  trControl = ctrl,
  tuneGrid = data.frame(maxdepth = seq(1, 15, 1))
)

end.time <- Sys.time()
print(end.time - start.time)

m_tree  # maxdepth = 2, cv = .7479
plot(varImp(m_tree), top = 5, main = "Decision Tree") 
#plot(m_tree, main = 'Decision Tree CV')
#write.csv(m_tree$results, "20201122_mod1_tree.csv")

rpart.plot(
  # the model with the best accuracy
  m_tree$finalModel, 
  # show % of obs
  extra = 100,
  yesno = 2)

cv_result <-
  cv_result %>%
  add_row(tibble(Model = 'Tree',
                 Param = 2,
                 cv = .7479))


##-------------------
## 3. Random Forest
##    Approx 59 min
##-------------------
set.seed(20)

start.time <- Sys.time() # time 

m_rf <- train(
  use~.,
  data = micsTrain,
  trControl = ctrl,
  method = "rf",
  importance = T,
  tuneGrid = data.frame(mtry = c(3, 9))
)

end.time <- Sys.time() # time
print(end.time - start.time)

m_rf # mtry = 3, .7667
print(m_rf$finalModel)

# Variable importance
importance(m_rf$finalModel, type = 1)

rf_imp <- data.frame(importance(m_rf$finalModel, type = 1)) %>%
  rownames_to_column(var = "Variable") %>% 
  slice_max(order_by = MeanDecreaseAccuracy, n = 5) %>% 
  mutate(Variable <- fct_reorder(Variable,MeanDecreaseAccuracy))

dotplot(Variable~MeanDecreaseAccuracy, rf_imp)

cv_result <-
  cv_result %>%
  add_row(tibble(Model = 'RandomForest',
                 Param = 3,
                 cv = .7667))


##--------------------------
##  4. Logistic Regression  
##--------------------------

set.seed(20)

glm1 <-
  use ~ age + edu + mstat + given_birth + country * wealth

glm2 <-
  use ~ poly(age, 2, raw = TRUE) + edu + mstat + given_birth + country * wealth

glm3 <-
  use ~ poly(age, 3, raw = TRUE) + edu + mstat + given_birth + country * wealth

glm4 <-
  use ~ age + edu + mstat + given_birth + country * residence

glm5 <-
  use ~ poly(age, 2, raw = TRUE) + edu + mstat + given_birth + country * residence

glm6 <-
  use ~ poly(age, 3, raw = TRUE) + edu + mstat + given_birth + country * residence

m_glm1 <- train(glm1, micsTrain, method = "glm", trControl = ctrl, family = "binomial")
m_glm1 #.7556

m_glm2 <- train(glm2, micsTrain, method = "glm", trControl = ctrl, family = "binomial")
m_glm2 #.7642

m_glm3 <- train(glm3, micsTrain, method = "glm", trControl = ctrl, family = "binomial")
m_glm3 #.7641

m_glm4 <- train(glm4, micsTrain, method = "glm", trControl = ctrl, family = "binomial")
m_glm4 #.7546

m_glm5 <- train(glm5, micsTrain, method = "glm", trControl = ctrl, family = "binomial")
m_glm5 #.7626

m_glm6 <- train(glm6, micsTrain, method = "glm", trControl = ctrl, family = "binomial")
m_glm6 #.7628

m_logit <- m_glm2 #Best
summary(m_logit)

cv_result <-
  cv_result %>%
  add_row(tibble(Model = 'Logit',
                 Param = 2,
                 cv = .7642))

##----------
##  5. GAM  
##----------
set.seed(20)

# Basis cubic
gam1 <- 
  use ~ bs(age) + edu + mstat + given_birth + country * wealth

# Cubic Spline with three knots on quartile
gam2 <- 
  use ~ bs(age, knots = c(44)) + edu + mstat + given_birth + country * wealth

# Natural cubic spline w/ three knots
gam3 <-
  use ~ ns(age, knots = c(44)) + edu + mstat + given_birth + country * wealth

m_gam1 <- train(gam1, micsTrain, method = "glm", trControl = ctrl, family = "binomial")
m_gam1 #.7641

m_gam2 <- train(gam2, micsTrain, method = "glm", trControl = ctrl, family = "binomial")
m_gam2 #.7643

m_gam3 <- train(gam3, micsTrain, method = "glm", trControl = ctrl, family = "binomial")
m_gam3 #.7638

m_gam <- m_gam2 # Best
summary(m_gam$finalModel)
varImp(m_gam, scale = F)
m_gam$finalModel$coefficients
cv_result <-
  cv_result %>%
  add_row(tibble(Model = 'GAM',
                 Param = 2,
                 cv = .7643))

##===============================================================
##                        Model Testing                        ==
##===============================================================

##----------------------
##  Compare All Models  
##----------------------
final_model <- resamples(
  list(
    KNN = m_knn,
    DecisionTree = m_tree,
    RandomForest = m_rf,
    LogisticRegression = m_logit,
    GAM = m_gam
  )
)
summary(final_model)

bwplot(final_model, metric = "Accuracy")
cv_result$cv

##-----------
##  Predict  
##-----------
test_table <- tibble(
  Prediction = character(), Reference = character(),
  n = integer(), Model = character()
)

pred_tree <- predict(m_tree, newdata = micsTest)
tbl_tree  <- confusionMatrix(pred_tree, micsTest$use)


pred_knn <- predict(m_knn, newdata = micsTest)
tbl_knn  <- confusionMatrix(pred_knn, micsTest$use)
tbl_knn
tbl_knn$byClass[1]

pred_logit <- predict(m_logit, newdata = micsTest)
tbl_logit  <- confusionMatrix(pred_logit, micsTest$use)
tbl_logit$byClass

pred_gam <- predict(m_gam, newdata = micsTest)
tbl_gam  <- confusionMatrix(pred_gam, micsTest$use)
tbl_gam$byClass[1]

pred_rf <- predict(m_rf, newdata = micsTest)
tbl_rf  <- confusionMatrix(pred_rf, micsTest$use)
tbl_rf$byClass[1]

# Review Test Accuracy vs CV Error
test_result <- tibble(
  Model = c("KNN", "Tree", "RandomForest", "Logit", "GAM"),
  Test = c(tbl_knn$overall[1],
           tbl_tree$overall[1],
           tbl_rf$overall[1],
           tbl_logit$overall[1],
           tbl_gam$overall[1])
) %>% right_join(cv_result, on = "Model")
test_result

# Export confusion matrix
test_table <- 
  test_table %>% 
  add_row(
    as_tibble(tbl_knn$table) %>%
      add_column(Model = "KNN")
  ) %>%
  add_row(
    as_tibble(tbl_tree$table) %>%
      add_column(Model = "Tree")
  ) %>%
  add_row(
    as_tibble(tbl_rf$table) %>%
      add_column(Model = "RandomForest")
  ) %>%
  add_row(
    as_tibble(tbl_logit$table) %>%
      add_column(Model = "Logit")
  ) %>%
  add_row(
    as_tibble(tbl_gam$table) %>%
      add_column(Model = "GAM")
  )

test_table <- test_table %>% 
  pivot_wider(names_from = Reference, values_from = n, names_prefix = "Ref_") %>% 
  relocate(Model)

table(micsTest$use)
test_table
write.csv(test_table, "20201128_confusionmatrix.csv")
options(digits=4)
# Performance
data.frame(knn = tbl_knn$byClass[1:4]) %>%
  add_column(data.frame(tree = tbl_tree$byClass[1:4])) %>% 
  add_column(data.frame(rf = tbl_rf$byClass[1:4])) %>% 
  add_column(data.frame(logit = tbl_logit$byClass[1:4])) %>% 
  add_column(data.frame(gam = tbl_gam$byClass[1:4])) %>%
  rownames_to_column(var = "Metric") 



##################################################################
##                          Playground                          ##
##################################################################


##-------------------------------------------
##  Visualise & Combine Variable Importance  
##-------------------------------------------

varImptbl <- 
  varImp(m_tree)$importance %>% 
  slice_max(order_by = Overall, n = 5) %>% 
  rownames_to_column(var = "Var") %>% 
  mutate(Model = "DecisionTree") %>%
  bind_rows(varImp(m_logit)$importance %>%
              slice_max(order_by = Overall, n = 5) %>%
              rownames_to_column(var = "Var") %>% mutate(Model = "Logit")) %>%
  bind_rows(varImp(m_gam)$importance %>%
              slice_max(order_by = Overall, n = 5) %>%
              rownames_to_column(var = "Var") %>% mutate(Model = "GAM")) %>%
  bind_rows(varImp(m_rf)$importance%>%
              slice_max(order_by = Using, n = 5) %>%
              rownames_to_column(var = "Var") %>%
              mutate(Overall = Using,
                     Using = NULL,
                     `Not Using` = NULL,
                     Model = 'RandomForest')
  )


varImptbl <- varImptbl %>%
  mutate(Model = as_factor(Model)) %>%
  mutate(Var = factor(Var,
                      levels = c(
                        "`bs(age, knots = c(44))2`",
                        "`poly(age, 2, raw = TRUE)2`",
                        "age",   
                        "countryLaos",
                        "countryMongolia",
                        "eduPrimary",
                        "given_birthNo",
                        "mstatFormer",
                        "mstatNever" 
                      ),
                      labels = c(
                        "Age cubic knot",
                        "Age poly",
                        "Age",
                        "Laos",
                        "Mongolia",
                        "Primary edu",
                        "Never give birth",
                        "Former married",
                        "Never married"
                      )))


dotplot(importance(m_rf$finalModel, type = 1))
panel.needle(
  vartbl$Var, vartbl$MeanDecreaseAccuracy)
varImpPlot(m_rf$finalModel)
list(levels(factor(varImptbl$Var)))
dotplot(Var~Overall | Model, data = varImptbl, xlab = "Importance")

