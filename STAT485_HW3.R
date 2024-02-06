## R commands for STAT 485, Hw3
## Spring Semester 2024
## Reference: http://homepages.math.uic.edu/~jyang06/stat485/stat485.html

# install.packages("pls")


# load R packages
library("ElemStatLearn")  # for prostate cancer dataset
library(pls)

# load the data set "prostate"
data(prostate)
table(prostate$train)
# FALSE  TRUE 
# 30    67

# generate 100 random partitions
Nsimu = 100
set.seed(615)                       # specify the intital random seed
train.index <- matrix(0, Nsimu, 97) # each row indicates one set of simulated training indices
for(i in 1:Nsimu) train.index[i,]=sample(x=c(rep(1,67),rep(0,30)), size=97, replace=F)   # generate random indices of training data


# define matrices to store results
APerror <- matrix(0, Nsimu, 4)      # mean (absolute) prediction error
SPerror <- matrix(0, Nsimu, 4)      # mean (squared) prediction error
colnames(APerror)=colnames(SPerror)=c("FullModel", "ReducedModel(4)", "PCR","PLS")
Tuning <- APerror                   # record values of tuning parameters

ttemp=proc.time()                   # record computing time
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
  fitls <- lm( lpsa ~ lcavol+lweight+age+lbph+svi+lcp+gleason+pgg45, data=trainst )
  fitlsr4 <- lm( lpsa ~ lcavol+lweight+lbph+svi, data=trainst )
  
  ## check testing errors
  testst <- test
  for(i in 1:8) {
    testst[,i] <- testst[,i] - mean(prostate[,i]);
    testst[,i] <- testst[,i]/sd(prostate[,i]);
  }
  
  ## (I) mean prediction error based on full model
  test.fitls=predict(fitls, newdata=testst)  
  # mean (absolute) prediction error
  APerror[isimu,1]=mean(abs(test[,9]-test.fitls))  
  # mean (squared) prediction error
  SPerror[isimu,1]=mean((test[,9]-test.fitls)^2)  
  
  ## (II) mean prediction error based on reduced model(4)
  test.fitlsr=predict(fitlsr4, newdata=testst)  
  # mean (absolute) prediction error
  APerror[isimu,2]=mean(abs(test[,9]-test.fitlsr))  
  # mean (squared) prediction error
  SPerror[isimu,2]=mean((test[,9]-test.fitlsr)^2)
  
  ## (III) Principal Component Analysis
  pcr.fit=pcr(lpsa~., data=trainst, scale=F, validation="CV", segments=10)
  # find the best number of components, regenerating part of Figure 3.7 on page 62
  itemp=which.min(pcr.fit$validation$PRESS)     
  itemp.mean=pcr.fit$validation$PRESS[itemp]/67 
  mean((pcr.fit$validation$pred[,,itemp]-trainst[,9])^2) 
  itemp.sd=sd((pcr.fit$validation$pred[,,itemp]-trainst[,9])^2)/sqrt(67)   
  k.pcr = min((1:pcr.fit$validation$ncomp)[pcr.fit$validation$PRESS/67 < itemp.mean+itemp.sd])  # the chosen k
  
  # estimating mean prediction error
  test.pcr=predict(pcr.fit,as.matrix(testst[,1:8]),ncomp=k.pcr)
  # mean (absolute) prediction error
  APerror[isimu,3] = mean(abs(test[,9]-test.pcr))                
  # mean (squared) prediction error
  SPerror[isimu,3] = mean((test[,9]-test.pcr)^2) 
  # Keep track of tuning parameter k
  Tuning[isimu, 3]= k.pcr
  

  ## (IV) Partial Least Squares
  plsr.fit=plsr(lpsa~., data=trainst, scale=F, validation="CV", segments=10)
  # find the best number of components
  itemp=which.min(plsr.fit$validation$PRESS)     
  itemp.mean=plsr.fit$validation$PRESS[itemp]/67
  mean((plsr.fit$validation$pred[,,itemp]-trainst[,9])^2) 
  itemp.sd=sd((plsr.fit$validation$pred[,,itemp]-trainst[,9])^2)/sqrt(67)  
  k.plsr = min((1:plsr.fit$validation$ncomp)[plsr.fit$validation$PRESS/67 < itemp.mean+itemp.sd])  # the chosen k
  
  # estimating mean prediction error
  test.plsr=predict(plsr.fit,as.matrix(testst[,1:8]),ncomp=k.plsr)
  # mean (absolute) prediction error
  APerror[isimu,4] = mean(abs(test[,9]-test.plsr))               
  # mean (squared) prediction error
  SPerror[isimu,4]  = mean((test[,9]-test.plsr)^2)                 
  # Keep track of tuning parameter k
  Tuning[isimu, 4]= k.plsr
         
}                                                               # end of loop with "isimu"
proc.time()-ttemp
# user  system elapsed 
# 1.24    0.07    1.34


## plot the output
par(mfrow=c(1,1))
# mean (absolute) prediction error
boxplot(APerror)
# mean (squared) prediction error
boxplot(SPerror)

## check summary statistics
# mean (absolute) prediction error
summary(APerror)

# FullModel      ReducedModel(4)       PCR              PLS        
# Min.   :0.3759   Min.   :0.3683   Min.   :0.4702   Min.   :0.4322  
# 1st Qu.:0.5198   1st Qu.:0.5211   1st Qu.:0.5918   1st Qu.:0.5913  
# Median :0.5724   Median :0.5641   Median :0.6526   Median :0.6295  
# Mean   :0.5745   Mean   :0.5702   Mean   :0.6529   Mean   :0.6362  
# 3rd Qu.:0.6327   3rd Qu.:0.6241   3rd Qu.:0.7049   3rd Qu.:0.6854  
# Max.   :0.7926   Max.   :0.7600   Max.   :1.0320   Max.   :0.9405

# mean (squared) prediction error
summary(SPerror)
# FullModel      ReducedModel(4)       PCR              PLS        
# Min.   :0.2711   Min.   :0.2440   Min.   :0.3357   Min.   :0.3295  
# 1st Qu.:0.4736   1st Qu.:0.4491   1st Qu.:0.5760   1st Qu.:0.5706  
# Median :0.5718   Median :0.5271   Median :0.6769   Median :0.6510  
# Mean   :0.5657   Mean   :0.5340   Mean   :0.6836   Mean   :0.6591  
# 3rd Qu.:0.6384   3rd Qu.:0.6117   3rd Qu.:0.7697   3rd Qu.:0.7355  
# Max.   :0.9210   Max.   :0.8540   Max.   :1.5136   Max.   :1.2708 

# test the difference on mean (absolute) prediction error
pvalue.APE <- matrix(1, 4, 4)# matrix with (i,j)-entry indicating p-values of corresponding paired t-tests on mean (absolute) predition error
rownames(pvalue.APE)=colnames(pvalue.APE)=c("FullModel", "ReducedModel(4)", "PCR","PLS")
pvalue.SPE <- pvalue.APE     # matrix with (i,j)-entry indicating p-values of corresponding paired t-tests on mean (squared) prediction error
for(i in 1:3) for(j in (i+1):4) {
  pvalue.APE[i,j]=pvalue.APE[j,i]=t.test(x=APerror[,i], y=APerror[,j], alternative="two.sided", paired=T)$p.value;
  pvalue.SPE[i,j]=pvalue.SPE[j,i]=t.test(x=SPerror[,i], y=SPerror[,j], alternative="two.sided", paired=T)$p.value;
}
round(pvalue.APE,3)

#                    FullModel ReducedModel(4) PCR PLS
# FullModel           1.000           0.234   0   0
# ReducedModel(4)     0.234           1.000   0   0
# PCR                 0.000           0.000   1   0
# PLS                 0.000           0.000   0   1


## median of APerror
round(apply(APerror,2,median),4)
# FullModel   ReducedModel(4)     PCR             PLS 
# 0.5724          0.5641          0.6526          0.6295

sort(apply(APerror,2,median))

# Alternative hypothesis: true difference in means is not equal to 0
# Conclusion: In terms of mean (absolute) prediction error, 
#   ReducedModel(4) <=  FullModel < PLS < PCR

round(pvalue.SPE,3)
#                FullModel ReducedModel(4)   PCR   PLS
#FullModel               1               0 0.000 0.000
#ReducedModel(4)         0               1 0.000 0.000
#PCR                     0               0 1.000 0.003
#PLS                     0               0 0.003 1.000

round(apply(SPerror,2,median),4)
# FullModel  ReducedModel(4)       PCR             PLS 
# 0.5718          0.5271          0.6769          0.6510 
# Alternative hypothesis: true difference in means is not equal to 0
# Conclusion: In terms of mean (squared) prediction error, 
#   ReducedModel(4) <  FullModel < PLS < PCR

## among the 100 random partitions, how many times one method is better than another in terms of mean (absolute) prediction error
count.better.AP <- matrix(0, 4, 4)  # entry (i,j): how many times method i is better than method j using mean (absolute) prediction error
rownames(count.better.AP)=colnames(count.better.AP)=c("FullModel", "ReducedModel(4)", "PCR","PLS")
count.better.SP <- count.better.AP  # entry (i,j): how many times method i is better than method j using mean (squared) prediction error
for(i in 1:4) for(j in 1:4) {
  count.better.AP[i,j] = sum(APerror[,i] < APerror[,j]);
  count.better.SP[i,j] = sum(SPerror[,i] < SPerror[,j]);
}
count.better.AP

#                 FullModel ReducedModel(4) PCR PLS
#FullModel               0              53  83  84
#ReducedModel(4)        47               0  94  87
#PCR                    17               6   0  33
#PLS                    16              13  67   0


count.better.SP

#                 FullModel ReducedModel(4) PCR PLS
#FullModel               0              33  81  78
#ReducedModel(4)        67               0  94  92
#PCR                    19               6   0  34
#PLS                    22               8  66   0

## Conclusion: The performance of PCR, PLS depends on the training/testing partition.
## check tuning parameters
summary(Tuning[,3:4])
#          PCR          PLS     
# Min.   :1.00   Min.   :1.0  
# 1st Qu.:3.00   1st Qu.:1.0  
# Median :3.00   Median :1.0  
# Mean   :3.04   Mean   :1.4  
# 3rd Qu.:3.00   3rd Qu.:2.0  
# Max.   :8.00   Max.   :3.0

## When PCR does not performance better
plot(Tuning[,3], SPerror[,2]-SPerror[,3], xlab="k", ylab="Errors: Reduced-PCR")
abline(0,0)
# Conclusion: Too large/small k leads to worse performance.

## When PLS does not performance better
plot(Tuning[,4], SPerror[,2]-SPerror[,4], xlab="k", ylab="Errors: Reduced-PLS")
abline(0,0)
# Conclusion: Too small k leads to worse performance.


