## R commands for STAT 485, Hw3
## Spring Semester 2024
## Reference: http://homepages.math.uic.edu/~jyang06/stat485/stat485.html

## Step 1: Install the KNN packages
#install.packages(knn)
#install.packages(FNN)
# In my R 4.3 version, knn doesn't support. Use FNN instead.


library(FNN)

# Step 2: Wrong procedure

set.seed(123) # For reproducibility

# Initialize variables
total_error_rate <- 0
n_simulations <- 100

for (i_sim in 1:n_simulations) {
  # Generate data
  y <- sample(c(rep(1,25), rep(2,25)), size=50, replace=FALSE)
  x <- matrix(rnorm(50 * 5000), nrow = 50, ncol = 5000)
  
  # Compute correlations and select top 100 predictors
  xcor <- cor(x, y)
  xind <- order(-abs(xcor))[1:100]
  x_selected <- x[, xind]
  
  # 5-fold Cross-validation
  fold_size <- 10
  error_rate <- 0
  
  for (i_fold in 1:5) {
    # Split data into training and test sets
    test_indices <- ((i_fold - 1) * fold_size + 1):(i_fold * fold_size)
    x_train <- x_selected[-test_indices, ]
    y_train <- y[-test_indices]
    x_test <- x_selected[test_indices, ]
    y_test <- y[test_indices]
    
    # Apply KNN
    knn_pred <- knn(train = x_train, test = x_test, cl = y_train, k = 1)
    
    # Calculate error rate for this fold
    fold_error_rate <- sum(knn_pred != y_test) / length(y_test)
    error_rate <- error_rate + fold_error_rate
  }
  
  # Average error rate for this simulation
  total_error_rate <- total_error_rate + (error_rate / 5)
}

# Average cross-validation error rate across all simulations
average_error_rate <- total_error_rate / n_simulations
print(average_error_rate)
# [1] 0.0114



# Step 3: 

set.seed(123) # For reproducibility

# Initialize variables
n_simulations <- 50
n_folds <- 5
n_samples <- 50
n_predictors <- 5000
n_top_predictors <- 100

average_error_rates <- numeric(n_simulations)

for (i_sim in 1:n_simulations) {
  # Generate data
  y <- sample(c(rep(1,25), rep(2,25)), size=n_samples, replace=FALSE)
  x <- matrix(rnorm(n_samples * n_predictors), nrow = n_samples, ncol = n_predictors)
  
  # Create folds
  folds <- cut(seq(1, n_samples), breaks=n_folds, labels=FALSE)
  
  total_error_rate <- 0
  
  for (i_fold in 1:n_folds) {
    # Split data into training and test sets
    test_indices <- which(folds == i_fold)
    train_indices <- setdiff(1:n_samples, test_indices)
    
    x_train <- x[train_indices, ]
    y_train <- y[train_indices]
    x_test <- x[test_indices, ]
    y_test <- y[test_indices]
    
    # Find top 100 predictors
    xcor <- cor(x_train, y_train)
    top_predictors <- order(-abs(xcor))[1:n_top_predictors]
    
    # Train and test using KNN
    knn_pred <- knn(train = x_train[, top_predictors], test = x_test[, top_predictors], cl = y_train, k = 1)
    
    # Calculate error rate for this fold
    fold_error_rate <- sum(knn_pred != y_test) / length(y_test)
    total_error_rate <- total_error_rate + fold_error_rate
  }
  
  # Average error rate for this simulation
  average_error_rates[i_sim] <- total_error_rate / n_folds
}

# Overall average cross-validation error rate
overall_average_error_rate <- mean(average_error_rates)
print(overall_average_error_rate)
# [1] 0.5072