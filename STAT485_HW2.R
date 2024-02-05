## R commands for STAT 485, Hw2

# load the package  "ElemStatLearn" 
library("ElemStatLearn")
## use glmnet in package glmnet
library(glmnet)
library(lars)

# load the data set "prostate"
data(prostate)
table(prostate$train)
# FALSE  TRUE 
# 30    67

# generate 100 random partitions
Nsimu = 100
set.seed(532)                       # specify the intital random seed
train.index <- matrix(0, Nsimu, 97) # each row indicates one set of simulated training indices
for(i in 1:Nsimu) train.index[i,]=sample(x=c(rep(1,67),rep(0,30)), size=97, replace=F)   # generate random indices of training data

# define matrices to store results
APerror <- matrix(0, Nsimu, 7)      # mean (absolute) prediction error
SPerror <- matrix(0, Nsimu, 7)      # mean (squared) prediction error

#(I) full linear model; (II) reduced linear model (4) with lcavol, lweight, lbph,svi; (III) reduced linear model (2) with lcavol, lweight; (IV) subset selection using
#R function step; (V) Ridge regression; (VI) Lasso; (VII) Lars.

colnames(APerror)=colnames(SPerror)=c("FullModel", "ReducedModel_4","ReducedModel_2","Step","Ridge","Lasso","Lars")


# iterate through the 100 partitions with the 7 models
for(isimu in 1:Nsimu) {             # start of loop with "isimu"
  # partition the original data into training and testing datasets
  train <- subset( prostate, train.index[isimu,]==1 )[,1:9]
  test  <- subset( prostate, train.index[isimu,]==0 )[,1:9]  
  # fit linear model on training dataset using LS method
  trainst <- train
  for(i in 1:8) {
    trainst[,i] <- trainst[,i] - mean(prostate[,i]);
    trainst[,i] <- trainst[,i]/sd(prostate[,i]);
  }
  
  # Full model
  fitls <- lm( lpsa ~ lcavol+lweight+age+lbph+svi+lcp+gleason+pgg45, data=trainst )
  
  # Reduced with 4 covariates
  fitlsr <- lm( lpsa ~ lcavol+lweight+lbph+svi, data=trainst )
  
  # Reduced with 2 covariates
  fitlsr2 <- lm( lpsa ~ lcavol+lweight, data=trainst ) 
  
  #Subset selection through step
  model.step=step(fitls)
  coef.step=model.step$coefficients
  k=length(coef.step)
  fitlsr.step <- lm( lpsa ~ ., data=trainst[,c(names(coef.step[2:k]),"lpsa")] )
  
  #Ridge regression
  # use 10-fold cross-validation to choose best lambda
  cv.out=cv.glmnet(x=as.matrix(trainst[,1:8]), y=as.numeric(trainst[,9]), nfolds=10, alpha=0, standardize=F)
  lambda.10fold=cv.out$lambda.1s   #Store the lambda for Ridge
  # apply Ridge regression with chosen lambda
  fitridge=glmnet(x=as.matrix(trainst[,1:8]),y=as.numeric(trainst[,9]),alpha=0,lambda=lambda.10fold,standardize=F,thresh=1e-12)
  
  # LASSO
  cv.out2=cv.glmnet(x=as.matrix(trainst[,1:8]), y=as.numeric(trainst[,9]), nfolds=10, alpha=1, standardize=F)
  lambda2.10fold=cv.out2$lambda2.1s
  # Apply LASSO with best lambda2
  fitlasso=glmnet(x=as.matrix(trainst[,1:8]),y=as.numeric(trainst[,9]),alpha=1,lambda=lambda2.10fold,standardize=F,thresh=1e-12)
  
  
  #LARS
  prostate.lar <- lars(x=as.matrix(trainst[,1:8]), y=as.numeric(trainst[,9]), type="lar", trace=TRUE, normalize=F)
  # choose k using 10-fold cross-validation, "cv.lars" also generates a graph similar to Figure 3.7 on page 62
  cv.out3 <- cv.lars(x=as.matrix(trainst[,1:8]), y=as.numeric(trainst[,9]), K=10, plot.it=T, type="lar", trace=TRUE, normalize=F)
  itemp=which.min(cv.out3$cv) # 8
  k.lars = min(cv.out3$index[cv.out3$cv < cv.out3$cv[itemp]+cv.out3$cv.error[itemp]]) 
  
  
  ## check testing errors
  testst <- test
  for(i in 1:8) {
    testst[,i] <- testst[,i] - mean(prostate[,i]);
    testst[,i] <- testst[,i]/sd(prostate[,i]);
  }
  
  # mean prediction error on testing data using mean training value
  
  # mean prediction error based on full model
  test.fitls=predict(fitls, newdata=testst)  
  # mean (absolute) prediction error
  APerror[isimu,1]=mean(abs(test[,9]-test.fitls))  
  # mean (squared) prediction error
  SPerror[isimu,1]=mean((test[,9]-test.fitls)^2)
  
  
  # mean prediction error based on reduced model
  test.fitlsr=predict(fitlsr, newdata=testst)  
  # mean (absolute) prediction error
  APerror[isimu,2]=mean(abs(test[,9]-test.fitlsr))  
  # mean (squared) prediction error
  SPerror[isimu,2]=mean((test[,9]-test.fitlsr)^2)
  
  # mean prediction error based on reduced model with 2 covariates
  test.fitlsr2=predict(fitlsr2, newdata=testst)  
  # mean (absolute) prediction error
  APerror[isimu,3]=mean(abs(test[,9]-test.fitlsr2))  
  # mean (squared) prediction error
  SPerror[isimu,3]=mean((test[,9]-test.fitlsr2)^2)
  
  # mean prediction error based on reduced model with step subset selection
  test.fitlsr.step=predict(fitlsr.step, newdata=testst)  
  # mean (absolute) prediction error
  APerror[isimu,4]=mean(abs(test[,9]-test.fitlsr.step))  
  # mean (squared) prediction error
  SPerror[isimu,4]=mean((test[,9]-test.fitlsr.step)^2)  
  
  # estimating mean prediction error
  test.ridge=predict(fitridge,newx=as.matrix(testst[,1:8]))
  # mean (absolute) prediction error
  APerror[isimu,5]=mean(abs(test[,9]-test.ridge))               
  # mean (squared) prediction error
  SPerror[isimu,5]=mean((test[,9]-test.ridge)^2)                
  
  # estimating mean prediction error
  test.lasso=predict(fitlasso,newx=as.matrix(testst[,1:8]))
  # mean (absolute) prediction error
  APerror[isimu,6]=mean(abs(test[,9]-test.lasso))                
  # mean (squared) prediction error
  SPerror[isimu,6]=mean((test[,9]-test.lasso)^2)               
  
  
  # estimating mean prediction error
  test.lars=predict(prostate.lar, newx=as.matrix(testst[,1:8]), s=k.lars, type="fit", mode=cv.out3$mode)$fit
  # mean (absolute) prediction error
  APerror[isimu,7]=mean(abs(test[,9]-test.lars))        
  # mean (squared) prediction error
  SPerror[isimu,7]=mean((test[,9]-test.lars)^2)         
  
  
}
# plot the output, we might wanna use this to answer Q(3)
cind <- matrix(0, Nsimu, 7)
for(i in 1:7) cind[,i]=i;
boxplot(APerror~cind)
boxplot(SPerror~cind)


# Q(1) Report all pairwise t-test result at 5%.

# Write a function to loop through the result matrix.
compare_methods <- function(data, level = 0.05) {
  # Number of methods
  num_methods <- ncol(data)
  
  # Initialize a matrix to store the results
  results_matrix <- matrix(NA, nrow = num_methods, ncol = num_methods)
  rownames(results_matrix) <- colnames(data)
  colnames(results_matrix) <- colnames(data)
  
  # Perform pairwise comparisons
  for (i in 1:(num_methods-1)) {
    for (j in (i+1):num_methods) {
      test_result <- t.test(data[,i], data[,j], alternative="less", paired=TRUE)
      # Determine whether to reject the null hypothesis
      if (test_result$p.value < level) {
        results_matrix[i, j] <- "Reject"
        results_matrix[j, i] <- "Reject"
      } else {
        results_matrix[i, j] <- "Do not Reject"
        results_matrix[j, i] <- "Do not Reject"
      }
    }
  }
  
  return(results_matrix)
}

result_matrix_APerror <- compare_methods(APerror)
print(result_matrix_APerror)


result_matrix_SPerror <- compare_methods(SPerror)
print(result_matrix_SPerror)



# Q(2) Does your conclusion change across different partitions of training/testing sets?

# Write a function to compare values within each row. Keep count of how many times unit value is larger than another.

compare_all_with_others <- function(APerror) {
  # Number of methods and observations
  num_methods <- ncol(APerror)
  num_obs <- nrow(APerror)
  
  # Initialize a matrix to store the counts
  count_matrix <- matrix(0, nrow = num_obs, ncol = num_methods)
  colnames(count_matrix) <- colnames(APerror)
  
  # Iterate over each element in the matrix
  for (i in 1:num_obs) {
    for (j in 1:num_methods) {
      # Compare the current value with others in the same row
      for (k in 1:num_methods) {
        if (j != k && APerror[i, j] > APerror[i, k]) {
          count_matrix[i, j] <- count_matrix[i, j] + 1
        }
      }
    }
  }
  
  return(count_matrix)
}


count_result_APerror <- compare_all_with_others(APerror)
print(count_result_APerror)


count_result_SPerror <- compare_all_with_others(SPerror)
print(count_result_SPerror)



# Q(3) No, we cannot conclude Ridge, LASSO and LARS is better. Actually their APerror and SPerror are larger. But they're helpful when we face high-demension datasets.
column_means <- colMeans(APerror)
column_means2 <- colMeans(SPerror)

print(column_means)
print(column_means2)
