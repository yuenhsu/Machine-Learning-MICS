##================================================================
##                          tidymodels                           =
##================================================================

library(tidymodels)
m <- readRDS("MICS.rds")

# Split
set.seed(20)
m_split <- initial_split(m, prop = .85)
train_data <- training(m_split)
test_data <- testing(m_split)

# 10-fold cv
train_cv <- vfold_cv(train_data, v = 10)

# Pre-process recipe
m_recipe_dummy <-
  recipe(use~., data = train_data) %>%
  step_dummy(all_nominal(), -all_outcomes())

m_recipe <-
  recipe(use~., data = train_data)

# Decision tree
# Set engine
tree_engine <-
  decision_tree(mode = "classification",
                cost_complexity = tune(),
                min_n = 20) %>%
  set_engine("rpart")

# Workflow combining pre-process and modelling
tree_workflow <-
  workflow() %>%
  add_recipe(m_recipe) %>%
  add_model(tree_engine)

# Parameter tuning
tree_tune <-
  tune_grid(tree_workflow,
            resamples = train_cv,
            grid = expand.grid(cost_complexity = c(.01, .1)),
            metrics = metric_set(accuracy))
collect_metrics(tree_tune)

# Select best parameters and re-fit
tree_best <-
  finalize_workflow(tree_workflow, select_best(tree_tune)) %>%
  fit(train_data)
tree_best %>% pull_workflow_spec()

# training prediction
tree_best %>%
  predict(train_data) %>%
  bind_cols(train_data) %>%
  metrics(truth = use, estimate = .pred_class)

# testing prediction
tree_best %>%
  predict(test_data) %>%
  bind_cols(test_data) %>%
  metrics(truth = use, estimate = .pred_class)
