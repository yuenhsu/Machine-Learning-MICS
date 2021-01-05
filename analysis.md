---
title: "Machine learning with MICS microdata"
subtitle: "Predicting the use of contraception"
author: "Yu-En"
date: "05/01/2021"
output: 
  html_document: 
    keep_md: yes
---



## Introduction

The purpose of the repository is to demonstrate how to use `caret` to build classification models. This document contains the scripts for machine learning models that aim to predict the use of contraception in Thailand, Laos, and Mongolia with data from the [Multiple Indicator Cluster Surveys (MICS)](https://mics.unicef.org/about) of the UNICEF. The datasets are individual survey responses, and user registrations are required to access the microdata. Therefore, the data are not included in this repository.

The goal is to predict `use`, a binary variable that indicates whether a women of reproductive age is currently using contraception. The project report is available [here](https://yuenhsu.medium.com/predicting-contraception-use-in-asia-with-machine-learning-algorithms-d6bfab783e8).

### How to obtain the data

1. Register an account and download datasets from the [Survey](http://mics.unicef.org/surveys) page
2. Unzip the file and locate the "wm" module/file

The World Bank's [Microdata Library](https://microdata.worldbank.org/index.php/catalog/MICS) provides detailed data description for variable and value labels. Thailand, Laos, and Mongolia were selected because they are all in the East Asia Pacific (EAP) region, in the 6th round, and have data available.

***

### Load packages and data


```r
library(caret) # machine learning
library(tidyverse) # data wrangling
library(haven) # read dta file
library(arsenal) # summary statistic table
library(rpart.plot) # plot decision tree
mics <- read_dta("MICS.dta")
```




```r
mics %>% glimpse()
```

```
## Rows: 58,356
## Columns: 9
## $ age         <dbl> 29, 35, 36, 24, 34, 38, 24, 16, 15, 36, 40, 26, 26, 39, 4…
## $ edu         <fct> Higher Education, Higher Education, Higher Education, Hig…
## $ mstat       <fct> Current, Current, Never, Never, Current, Current, Never, …
## $ wealth      <dbl> 9, 9, 5, 8, 10, 4, 10, 7, 7, 8, 8, 10, 10, 9, 10, 10, 2, …
## $ residence   <fct> Urban, Urban, Urban, Urban, Urban, Urban, Urban, Urban, U…
## $ country     <fct> Thailand, Thailand, Thailand, Thailand, Thailand, Thailan…
## $ given_birth <fct> Yes, Yes, No, No, Yes, Yes, No, No, No, Yes, No, No, No, …
## $ child_died  <fct> No, No, No, No, No, No, No, No, No, No, No, No, No, No, N…
## $ use         <fct> Using, Using, Not Using, Not Using, Using, Using, Not Usi…
```

There are 9 predictors. All categorical features have been factorised. Please note that `wealth` is not the monetary value of household assets but an index^1^ that ranges from 1 to 10. It is country-specific and must be paired with `country`.

#### Descriptive Statistics


```r
tableby(use~., 
        data = mics,
        control = tableby.control(numeric.stats = c("mean","iqr","median"),
                                  cat.stats = c("countrowpct"),
                                  test = FALSE)) %>%
  summary()
```



|                                    | Using (N=29025) | Not Using (N=29331) | Total (N=58356) |
|:-----------------------------------|:---------------:|:-------------------:|:---------------:|
|**respondent's age**                |                 |                     |                 |
|&nbsp;&nbsp;&nbsp;Mean              |     34.432      |       29.562        |     31.984      |
|&nbsp;&nbsp;&nbsp;IQR               |     13.000      |       20.000        |     16.000      |
|&nbsp;&nbsp;&nbsp;Median            |     35.000      |       28.000        |     32.000      |
|**edu**                             |                 |                     |                 |
|&nbsp;&nbsp;&nbsp;Less than Primary |  2653 (48.8%)   |    2782 (51.2%)     |  5435 (100.0%)  |
|&nbsp;&nbsp;&nbsp;Primary           |  9023 (60.1%)   |    5998 (39.9%)     | 15021 (100.0%)  |
|&nbsp;&nbsp;&nbsp;Lower Secondary   |  6114 (49.4%)   |    6271 (50.6%)     | 12385 (100.0%)  |
|&nbsp;&nbsp;&nbsp;Upper Secondary   |  5080 (42.4%)   |    6900 (57.6%)     | 11980 (100.0%)  |
|&nbsp;&nbsp;&nbsp;Higher Education  |  6155 (45.5%)   |    7380 (54.5%)     | 13535 (100.0%)  |
|**mstat**                           |                 |                     |                 |
|&nbsp;&nbsp;&nbsp;Current           |  27773 (66.1%)  |    14229 (33.9%)    | 42002 (100.0%)  |
|&nbsp;&nbsp;&nbsp;Former            |   915 (23.1%)   |    3040 (76.9%)     |  3955 (100.0%)  |
|&nbsp;&nbsp;&nbsp;Never             |   337 (2.7%)    |    12062 (97.3%)    | 12399 (100.0%)  |
|**wealth**                          |                 |                     |                 |
|&nbsp;&nbsp;&nbsp;Mean              |      5.234      |        5.275        |      5.255      |
|&nbsp;&nbsp;&nbsp;IQR               |      5.000      |        5.000        |      5.000      |
|&nbsp;&nbsp;&nbsp;Median            |      5.000      |        5.000        |      5.000      |
|**residence**                       |                 |                     |                 |
|&nbsp;&nbsp;&nbsp;Urban             |  10675 (47.2%)  |    11965 (52.8%)    | 22640 (100.0%)  |
|&nbsp;&nbsp;&nbsp;Rural             |  18350 (51.4%)  |    17366 (48.6%)    | 35716 (100.0%)  |
|**country**                         |                 |                     |                 |
|&nbsp;&nbsp;&nbsp;Thailand          |  13840 (57.1%)  |    10384 (42.9%)    | 24224 (100.0%)  |
|&nbsp;&nbsp;&nbsp;Laos              |  10766 (44.8%)  |    13259 (55.2%)    | 24025 (100.0%)  |
|&nbsp;&nbsp;&nbsp;Mongolia          |  4419 (43.7%)   |    5688 (56.3%)     | 10107 (100.0%)  |
|**given_birth**                     |                 |                     |                 |
|&nbsp;&nbsp;&nbsp;Yes               |  27982 (64.4%)  |    15455 (35.6%)    | 43437 (100.0%)  |
|&nbsp;&nbsp;&nbsp;No                |   1043 (7.0%)   |    13876 (93.0%)    | 14919 (100.0%)  |
|**child_died**                      |                 |                     |                 |
|&nbsp;&nbsp;&nbsp;Yes               |  2262 (55.2%)   |    1833 (44.8%)     |  4095 (100.0%)  |
|&nbsp;&nbsp;&nbsp;No                |  26763 (49.3%)  |    27498 (50.7%)    | 54261 (100.0%)  |

Out of 58,356 samples, 49.74% use contraception, and 50.26% do not. The median age is 35 and 28, respectively. There are considerable differences in contraceptive use by marital status and by experience giving birth. Only 2.7% and 23.1% of never and formerly married women use contraception, while 66.1% currently married individuals does. Women who have never given birth have a significantly lower prevalence of contraceptive use (7%) than those with experience giving birth (64.4%).

***

### Prepare data

Here, data preparation involves splitting training and testing set and setting up cross-validation. Typically, data preparation involves much more sophisticated steps in real-life application, such as dealing with missing data, selecting variables, engineering features, scaling and centering variables, etc. However, since this was my first time building models from the beginning to the end, I did not perform said steps. 

#### Split training and testing set

I set aside 15% of observations for the testing set, which is reserved for the final testing once the models are trained and optimised. The rest of 85% is used to develop classification models. Because I will experiment with different parameters, I also use 10-fold cross-validation on the training set to evaluate performance.


```r
set.seed(20)
# Split data into testing and training according to the outcome
train_index <- createDataPartition(mics$use,
                                   # 85% for training
                                   p = .85,
                                   times = 1,
                                   list = FALSE)

micsTrain <- mics[ train_index, ]
micsTest  <- mics[-train_index, ]

# Check target distribution
prop.table(table(mics$use)) # all
## 
##     Using Not Using 
## 0.4973782 0.5026218
prop.table(table(micsTrain$use)) # training set
## 
##     Using Not Using 
## 0.4973792 0.5026208
prop.table(table(micsTest$use)) # testing set
## 
##     Using Not Using 
##  0.497372  0.502628
```

#### Create 10-fold cross-validation (cv)

Typically, a simple `trainControl(method="cv", k=10)` would suffice, but the result may be different every time the command is executed. While `trainControl` provides a seed parameter for reproducibility, I had trouble setting it up and decided to use `createFolds`.


```r
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
cv_result <- tibble(model = character(), cv_accuracy = numeric())
```

***

### Build classification models

#### K-nearest neighbour


```r
set.seed(20)
m_knn <- train(form = use~.,
               data = micsTrain,
               method = 'knn',
               trControl = ctrl, 
               tuneLength = 10)
print(m_knn)
```

```
## k-Nearest Neighbors 
## 
## 49604 samples
##     8 predictor
##     2 classes: 'Using', 'Not Using' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 44644, 44644, 44644, 44644, 44643, 44643, ... 
## Resampling results across tuning parameters:
## 
##   k   Accuracy   Kappa    
##    5  0.7239738  0.4486773
##    7  0.7249414  0.4506758
##    9  0.7256873  0.4521989
##   11  0.7244979  0.4498616
##   13  0.7245382  0.4499760
##   15  0.7232883  0.4475032
##   17  0.7228448  0.4466399
##   19  0.7225424  0.4460655
##   21  0.7215747  0.4441569
##   23  0.7214739  0.4439750
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was k = 9.
```

A list of integer, from 5 to 23 with an interval of two, was provided for k. The result indicates that the model using nine neighbours (k = 9) has the lowest 10-fold cross-validation error rate of 25.78 per cent.


```r
# Store result
cv_result <- cv_result %>% 
  add_row(model="KNN", 
          cv_accuracy = m_knn$results %>% 
            slice_max(order_by = Accuracy, n=1) %>% 
            pull(Accuracy))
# Plot cv accuracy
plot(m_knn, main = "KNN 10-fold Cross-Validation")
```

![](analysis_files/figure-html/knn_result-1.png)<!-- -->



#### Decision tree


```r
set.seed(20)
m_tree <- train(
  use ~.,
  micsTrain,
  method = "rpart2",
  trControl = ctrl,
  tuneGrid = data.frame(maxdepth = seq(1, 15, 1))
)
print(m_tree)
```

```
## CART 
## 
## 49604 samples
##     8 predictor
##     2 classes: 'Using', 'Not Using' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 44644, 44644, 44644, 44644, 44643, 44643, ... 
## Resampling results across tuning parameters:
## 
##   maxdepth  Accuracy   Kappa    
##    1        0.7155871  0.4326353
##    2        0.7479032  0.4967668
##    3        0.7479032  0.4967668
##    4        0.7479032  0.4967668
##    5        0.7479032  0.4967668
##    6        0.7479032  0.4967668
##    7        0.7479032  0.4967668
##    8        0.7479032  0.4967668
##    9        0.7479032  0.4967668
##   10        0.7479032  0.4967668
##   11        0.7479032  0.4967668
##   12        0.7479032  0.4967668
##   13        0.7479032  0.4967668
##   14        0.7479032  0.4967668
##   15        0.7479032  0.4967668
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was maxdepth = 2.
```

Cross-validation results for ten maxdepth parameters are provided below. After two, increasing tree depth does not yield improvements in error rate. The best tree (maxdepth = 2) has two splits, one for the indicator for ever given birth and one for marital status.


```r
# Store result
cv_result <- cv_result %>% 
  add_row(model="Decision Tree", 
          cv_accuracy = m_tree$results %>% 
            slice_max(order_by = Accuracy, n=1) %>% 
            pull(Accuracy))
# Plot cv accuracy
plot(m_tree, main = 'Decision Tree CV')
```

![](analysis_files/figure-html/tree_result-1.png)<!-- -->

```r
# Plot variable importance
plot(varImp(m_tree), top = 5, main = "Decision Tree")
```

![](analysis_files/figure-html/tree_result-2.png)<!-- -->

According to the model, women who have never given birth are predicted to be not using contraception. For people who had experience giving birth, if they were formerly married or in a union, the outcome is not using contraception. If they were currently or never married, the model predicted them to be using methods to avoid pregnancy. The cross-validation error rate is 25.21 per cent.


```r
rpart.plot(
  # the model with the best accuracy
  m_tree$finalModel, 
  # show % of obs
  extra = 100,
  yesno = 2)
```

![](analysis_files/figure-html/tree_plot-1.png)<!-- -->



#### Random forest

The mtry parameter is nine for bagging and three for random forest. Both models have 500 trees. 


```r
set.seed(20)
m_rf <- train(
  use~.,
  data = micsTrain,
  trControl = ctrl,
  method = "rf",
  importance = T,
  tuneGrid = data.frame(mtry = c(3, 9))
)
print(m_rf)
```

```
## Random Forest 
## 
## 49604 samples
##     8 predictor
##     2 classes: 'Using', 'Not Using' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 44644, 44644, 44644, 44644, 44643, 44643, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##   3     0.7632851  0.5272613
##   9     0.7448591  0.4902077
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 3.
```

The 10-fold cross-validation error rate is 25.60% for bagging and 23.33% for random forest; therefore, random forest with mtry of three is selected. The following figure presents the importance measures, the mean decrease in accuracy, for the five most important variables.


```r
# Store result
cv_result <- cv_result %>% 
  add_row(model="Random Forest", 
          cv_accuracy = m_rf$results %>% 
            slice_max(order_by = Accuracy, n=1) %>% 
            pull(Accuracy))
# Variable importance
plot(varImp(m_rf), top = 5, main = "Random Forest")
```

![](analysis_files/figure-html/rf_result-1.png)<!-- -->


***

### Compare training performance


```r
model_list <- resamples(
  list(
    KNN = m_knn,
    DecisionTree = m_tree,
    RandomForest = m_rf
  )
)
bwplot(model_list, metric = "Accuracy")
```

![](analysis_files/figure-html/compare_model-1.png)<!-- -->

Using the validation set, which contains 8,752 observations, each model is tested with unseen data.


```r
pred_knn <- predict(m_knn, newdata = micsTest)
postResample(pred_knn, micsTest$use) # KNN
##  Accuracy     Kappa 
## 0.7265768 0.4540008
pred_tree <- predict(m_tree, newdata = micsTest)
postResample(pred_tree, micsTest$use) # Decision tree
##  Accuracy     Kappa 
## 0.7586837 0.5182810
pred_rf <- predict(m_rf, newdata = micsTest)
postResample(pred_rf, micsTest$use) 
##  Accuracy     Kappa 
## 0.7709095 0.5424751
```

***

### Conclusion

Random forest has the highest accuracy of 77.56%. Decision tree scores the highest in sensitivity and negative predictive value (NPV) at 94.14% and 90.88%, respectively. That is, when a WRA is using contraception, there is a 94.14% probability that decision tree will label her as using. And when an individual is predicted to be not using contraception by the decision tree, there is a 90.88% probability that she is not using any methods to avoid pregnancy. 


***

### Reference
^1^ The DHS Program. (2016). Wealth Index. The DHS Program. https://dhsprogram.com/topics/wealth-index/
