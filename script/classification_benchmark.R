# ==============================================================================
# Project: EMNIST Handwritten Letter Classification
# Script: Model Benchmarking & Dimensionality Reduction
# ==============================================================================


#load the libraries
if (!require("pacman")) install.packages("pacman")
pacman::p_load(here, MASS, class, nnet, xgboost, HDclassif, tidyverse)

#load the data
load(here("data", "task1.Rdata"))


err2<-function(observed,predicted)
{tab<-table(observed,predicted)
err<-1-sum(diag(tab))/sum(tab)
return(err)
}


# ------------------------------------------------------------------------------
# Step 1: Dimensionality Reduction (PCA)
# ------------------------------------------------------------------------------

##scenario 1

#conduct PCA with the covariance matrix of the training data with centered variables
prcomp.out.s1<-prcomp(train.data.s1)

#total variance of s1
totvar.s1<- sum(apply(train.data.s1,2,stats::var))
#cumulative proportion of explained variance by components
propvar.s1<-prcomp.out.s1$sdev^2/totvar.s1
cumpropvar.s1<-cumsum(propvar.s1)
ncomp.s1 <- which(cumpropvar.s1 >= 0.90)[1]
print(ncomp.s1)
#the first 69 components capture at least 90% of the variance in s1

##scenario 2

#conduct PCA with the covariance matrix of the training data with centered variables
prcomp.out.s2<-prcomp(train.data.s2)

#total variance of s2
totvar.s2<- sum(apply(train.data.s2,2,stats::var))
#cumulative proportion of explained variance by components
propvar.s2<-prcomp.out.s2$sdev^2/totvar.s2
cumpropvar.s2<-cumsum(propvar.s2)
ncomp.s2 <- which(cumpropvar.s2 >= 0.90)[1]
print(ncomp.s2)
#the first 65 components capture at least 90% of the variance in s2


# ------------------------------------------------------------------------------
# Step 2: Model Training & Evaluation
# ------------------------------------------------------------------------------

##LDA scenario 1

#compute unstandardized principal components
train.comp.s1<-as.matrix(train.data.s1)%*%prcomp.out.s1$rotation[,1:ncomp.s1]

#compute lda on principal components
ldacomp.out.s1<-lda(train.comp.s1,train.target.s1)

#training error
predlda.train.s1<-predict(ldacomp.out.s1,train.comp.s1)
tab<-table(train.target.s1,predlda.train.s1$class)
lda.train.error.s1 <- 1-sum(diag(tab))/sum(tab)
print(lda.train.error.s1)

#test error
#compute unstandardized components for test set, using rotation vectors of training data
test.comp.s1<-as.matrix(test.data)%*%prcomp.out.s1$rotation[,1:ncomp.s1]
predlda.test.s1<-predict(ldacomp.out.s1,test.comp.s1)
tab<-table(test.target,predlda.test.s1$class)
lda.test.error.s1 <- 1-sum(diag(tab))/sum(tab)
print(lda.test.error.s1)

##LDA scenario 2

#compute unstandardized principal components
train.comp.s2<-as.matrix(train.data.s2)%*%prcomp.out.s2$rotation[,1:ncomp.s2]

#compute lda on principal components
ldacomp.out.s2<-lda(train.comp.s2,train.target.s2)

#training error
predlda.train.s2<-predict(ldacomp.out.s2,train.comp.s2)
tab<-table(train.target.s2,predlda.train.s2$class)
lda.train.error.s2 <- 1-sum(diag(tab))/sum(tab)
print(lda.train.error.s2)

#test error
#compute unstandardized components for test set, using rotation vectors of training data
test.comp.s2<-as.matrix(test.data)%*%prcomp.out.s2$rotation[,1:ncomp.s2]
predlda.test.s2<-predict(ldacomp.out.s2,test.comp.s2)
tab<-table(test.target,predlda.test.s2$class)
lda.test.error.s2 <- 1-sum(diag(tab))/sum(tab)
print(lda.test.error.s2)

##QDA scenario 1

#compute qda on the first 69 principal components
qdacomp.out.s1<-qda(train.comp.s1,train.target.s1)

#training error
predqda.train.s1<-predict(qdacomp.out.s1,train.comp.s1)
tab<-table(train.target.s1,predqda.train.s1$class)
qda.train.error.s1 <- 1-sum(diag(tab))/sum(tab)
print(qda.train.error.s1)

#test error
predqda.test.s1<-predict(qdacomp.out.s1,test.comp.s1)
tab<-table(test.target,predqda.test.s1$class)
qda.test.error.s1 <- 1-sum(diag(tab))/sum(tab)
print(qda.test.error.s1)


##QDA scenario 2

#compute qda on the first 65 principal components
qdacomp.out.s2<-qda(train.comp.s2,train.target.s2)

#training error
predqda.train.s2<-predict(qdacomp.out.s2,train.comp.s2)
tab<-table(train.target.s2,predqda.train.s2$class)
qda.train.error.s2 <- 1-sum(diag(tab))/sum(tab)
print(qda.train.error.s2)

#test error
predqda.test.s2<-predict(qdacomp.out.s2,test.comp.s2)
tab<-table(test.target,predqda.test.s2$class)
qda.test.error.s2 <- 1-sum(diag(tab))/sum(tab)
print(qda.test.error.s2)

##KNN scenario 1

#knn() -> select tuning parameter K so that the test error is as low as possible
knnmax<-100
err<-matrix(rep(0,knnmax*2),nrow=knnmax)
set.seed(1)
for (j in 1:knnmax){
  predknn.train.s1<- knn(train.comp.s1, train.comp.s1, train.target.s1, k=j)
  err[j,1]<-err2(train.target.s1,predknn.train.s1)
  predknn.test.s1<- knn(train.comp.s1, test.comp.s1, train.target.s1, k=j)
  err[j,2]<-err2(test.target,predknn.test.s1)}

plot(-1,-1,xlim=c(1,100),ylim=c(0,0.15),xlab="k",ylab="error rate")
lines(c(1:100),err[,1],col="red")
lines(c(1:100),err[,2],col="blue")
legend("bottomright",c("training error", "test error"),lty=c(1,1),col=c("red","blue"))
ksel.s1<-which.min(err[,2])
ksel.s1

#training error
knn.train.error.s1<-err[ksel.s1,1]
knn.train.error.s1

#test error
knn.test.error.s1<-err[ksel.s1,2]
knn.test.error.s1

##KNN scenario 2

#knn() -> select tuning parameter K so that the test error is as low as possible
err<-matrix(rep(0,knnmax*2),nrow=knnmax)
set.seed(1)
for (j in 1:knnmax){
  predknn.train.s2<- knn(train.comp.s2, train.comp.s2, train.target.s2, k=j)
  err[j,1]<-err2(train.target.s2,predknn.train.s2)
  predknn.test.s2<- knn(train.comp.s2, test.comp.s2, train.target.s2, k=j)
  err[j,2]<-err2(test.target,predknn.test.s2)}

plot(-1,-1,xlim=c(1,100),ylim=c(0,0.25),xlab="k",ylab="error rate")
lines(c(1:100),err[,1],col="red")
lines(c(1:100),err[,2],col="blue")
legend("bottomright",c("training error", "test error"),lty=c(1,1),col=c("red","blue"))
ksel.s2<-which.min(err[,2])
ksel.s2

#training error
knn.train.error.s2<-err[ksel.s2,1]
knn.train.error.s2

#test error
knn.test.error.s2<-err[ksel.s2,2]
knn.test.error.s2


##Multinomial logistic regression (PC) scenario 1

#data frame to work in multinom()
d.s1<-data.frame(train.target.s1,train.comp.s1)

#run multinomial logistic regression on components
set.seed(1)
mlogist.s1<-multinom(train.target.s1~.,data=d.s1,maxit=1000)
summary(mlogist.s1)

#training error
predmlogist.train.s1<-predict(mlogist.s1,train.comp.s1)
tab<-table(d.s1$train.target.s1,predmlogist.train.s1)
mlr.train.error.s1 <- 1-sum(diag(tab))/sum(tab)
print(mlr.train.error.s1)

#test error
predmlogist.test.s1<-predict(mlogist.s1,test.comp.s1)
tab<-table(test.target,predmlogist.test.s1)
mlr.test.error.s1 <- 1-sum(diag(tab))/sum(tab)
print(mlr.test.error.s1)

##Multinomial logistic regression (PC) scenario 2

d.s2<-data.frame(train.target.s2,train.comp.s2)

#run multinomial logistic regression on components
set.seed(1)
mlogist.s2<-multinom(train.target.s2~.,data=d.s2,maxit=1000)
summary(mlogist.s2)

#training error
predmlogist.train.s2<-predict(mlogist.s2,train.comp.s2)
tab<-table(d.s2$train.target.s2,predmlogist.train.s2)
mlr.train.error.s2 <- 1-sum(diag(tab))/sum(tab)
print(mlr.train.error.s2)

#test error
predmlogist.test.s2<-predict(mlogist.s2,test.comp.s2)
tab<-table(test.target,predmlogist.test.s2)
mlr.test.error.s2 <- 1-sum(diag(tab))/sum(tab)
print(mlr.test.error.s2)

##Multinomial logistic regression (PC+PC^2) scenario 1

#squared unstandardized principal components training
train.comp.sq.s1 <- train.comp.s1^2
colnames(train.comp.sq.s1) <- paste0(colnames(train.comp.s1), "^2")
d.sq.s1<-data.frame(train.target.s1,train.comp.s1,train.comp.sq.s1)
#squared unstandardized principal components test
test.comp.sq.s1 <- test.comp.s1^2
colnames(test.comp.sq.s1) <- paste0(colnames(test.comp.s1), "^2")
d.test.sq.s1<-data.frame(test.comp.s1,test.comp.sq.s1)

#run multinomial logistic regression on components and squared components
set.seed(1)
mlogist<-multinom(train.target.s1~.,data=d.sq.s1,maxit=2000)
summary(mlogist)

#training error
predmlogist.train.sq.s1<-predict(mlogist,d.sq.s1)
tab<-table(train.target.s1,predmlogist.train.sq.s1)
mlr.sq.train.error.s1 <- 1-sum(diag(tab))/sum(tab)
print(mlr.sq.train.error.s1)

#test error
predmlogist.test.sq.s1<-predict(mlogist,d.test.sq.s1)
tab<-table(test.target,predmlogist.test.sq.s1)
mlr.sq.test.error.s1 <- 1-sum(diag(tab))/sum(tab)
print(mlr.sq.test.error.s1)

#Multinomial logistic regression (PC+PC^2) s2

train.comp.sq.s2 <- train.comp.s2^2
colnames(train.comp.sq.s2) <- paste0(colnames(train.comp.s2), "^2")
d.sq.s2<-data.frame(train.target.s2,train.comp.s2,train.comp.sq.s2)
test.comp.sq.s2 <- test.comp.s2^2
colnames(test.comp.sq.s2) <- paste0(colnames(test.comp.s2), "^2")
d.test.sq.s2<-data.frame(test.comp.s2,test.comp.sq.s2)

#run multinomial logistic regression on components and squared components
set.seed(1)
mlogist<-multinom(train.target.s2~.,data=d.sq.s2,maxit=1000)
summary(mlogist)

#training error
predmlogist.train.sq.s2<-predict(mlogist,d.sq.s2)
tab<-table(train.target.s2,predmlogist.train.sq.s2)
mlr.sq.train.error.s2 <- 1-sum(diag(tab))/sum(tab)
print(mlr.sq.train.error.s2)

#test error
predmlogist.test.sq.s2<-predict(mlogist,d.test.sq.s2)
tab<-table(test.target,predmlogist.test.sq.s2)
mlr.sq.test.error.s2 <- 1-sum(diag(tab))/sum(tab)
print(mlr.sq.test.error.s2)

#Gradient boosting scenario 1
#xgb.cv() xgb.train() -> select meaningful tuning parameters

#xgboost on first 69 principal components
#make data set for analysis with xgboost
#data set with predictors should be a matrix (not a data.frame)
#vector with target should contain values 0,1,2..
#convert to 0-based indexing
train.target.s1.xgb <- as.numeric(train.target.s1) - 1  
test.target.xgb <- as.numeric(test.target) - 1      
dtrain.s1<-xgb.DMatrix(data = as.matrix(train.comp.s1), label = train.target.s1.xgb)
dtest.s1<-xgb.DMatrix(data = as.matrix(test.comp.s1), label = test.target.xgb)

#specify parameters
#for multinomial logit using objective=multi:softprob will output predicted class probabilities
#for multinomial logit using eval_metric will minimize negative of loglikelihood
params <- list(
  objective = "multi:softprob",  # outputs class probabilities
  eval_metric = "mlogloss",      # multiclass log-loss
  num_class = 4,                #number of classes dependent variable
  eta = 0.05,                     # learning rate
  max_depth = 6,                 #maximum depth tree
  subsample = 0.8,               #proportion of training set used to grow trees
  colsample_bytree = 0.5         #proportion of predictors used to grow trees
)

set.seed(1)
#cross validation
cv <- xgb.cv(
  params = params,
  data = dtrain.s1,
  nrounds = 1000,                # max number of boosting rounds
  nfold = 10,                    # number of CV folds
  early_stopping_rounds = 20,   # stop if no improvement for 20 rounds
  verbose = 1
)


best_nrounds.s1 <- cv$early_stop$best_iteration
cat("Best number of trees:", best_nrounds.s1, "\n")

#estimate final model
final_model.s1 <- xgb.train(
  params = params,
  data = dtrain.s1,
  nrounds = best_nrounds.s1
)

#training error
pred_matrix <- predict(final_model.s1, newdata=train.comp.s1)
class.train<-apply(pred_matrix,1,which.max)-1
gb.train.error.s1 <- err2(train.target.s1.xgb,class.train)
print(gb.train.error.s1)


#test error
pred_matrix <- predict(final_model.s1, newdata=test.comp.s1)
class.test<-apply(pred_matrix,1,which.max)-1
gb.test.error.s1 <- err2(test.target.xgb,class.test)
print(gb.test.error.s1)

#Gradient boosting scenario 2

#xgboost on first 65 principal components
#make data set for analysis with xgboost
#data set with predictors should be a matrix (not a data.frame)
#vector with target should contain values 0,1,2..
#convert to 0-based indexing
train.target.s2.xgb <- as.numeric(train.target.s2) - 1  
test.target.xgb <- as.numeric(test.target) - 1  
dtrain.s2<-xgb.DMatrix(data = as.matrix(train.comp.s2), label = train.target.s2.xgb)
dtest.s2<-xgb.DMatrix(data = as.matrix(test.comp.s2), label = test.target.xgb)

#specify parameters
#for multinomial logit using objective=multi:softprob will output predicted class probabilities
#for multinomial logit using eval_metric will minimize negative of loglikelihood
params <- list(
  objective = "multi:softprob",  # outputs class probabilities
  eval_metric = "mlogloss",      # multiclass log-loss
  num_class = 4,                #number of classes dependent variable
  eta = 0.05,                     # learning rate
  max_depth = 6,                 #maximum depth tree
  subsample = 0.8,               #proportion of training set used to grow trees
  colsample_bytree = 0.5         #proportion of predictors used to grow trees
)

set.seed(1)
#cross validation
cv <- xgb.cv(
  params = params,
  data = dtrain.s2,
  nrounds = 1000,                # max number of boosting rounds
  nfold = 10,                    # number of CV folds
  early_stopping_rounds = 20,   # stop if no improvement for 20 rounds
  verbose = 1
)


best_nrounds.s2 <- cv$early_stop$best_iteration
cat("Best number of trees:", best_nrounds.s2, "\n")

#estimate final model
final_model.s2 <- xgb.train(
  params = params,
  data = dtrain.s2,
  nrounds = best_nrounds.s2
)

#training error
pred_matrix <- predict(final_model.s2, newdata=train.comp.s2)
class.train<-apply(pred_matrix,1,which.max)-1
gb.train.error.s2 <- err2(train.target.s2.xgb,class.train)
print(gb.train.error.s2)


#test error
pred_matrix <- predict(final_model.s2, newdata=test.comp.s2)
class.test<-apply(pred_matrix,1,which.max)-1
gb.test.error.s2 <- err2(test.target.xgb,class.test)
print(gb.test.error.s2)

#HDDA scenario 1

#center data
ctrain.data.s1<-scale(train.data.s1,center=TRUE,scale=FALSE)
ctest.data.s1<-scale(test.data,center=attr(ctrain.data.s1, "scaled:center"),scale=FALSE)

set.seed(1)
hdda.out.s1 <- hdda(ctrain.data.s1, train.target.s1, model="AKJBKQKD", d_select = "Cattell", threshold = 0.05)

#train error
pred.train.s1<-predict(hdda.out.s1, ctrain.data.s1, train.target.s1)
tab<-table(train.target.s1,pred.train.s1$class)
hdda.train.error.s1 <- 1-sum(diag(tab))/sum(tab)
print(hdda.train.error.s1)

#test error
pred.test.s1<-predict(hdda.out.s1, ctest.data.s1, test.target)
tab<-table(test.target,pred.test.s1$class)
hdda.test.error.s1 <- 1-sum(diag(tab))/sum(tab)
print(hdda.test.error.s1)

#HDDA scenario 2

#center data
ctrain.data.s2<-scale(train.data.s2,center=TRUE,scale=FALSE)
ctest.data.s2<-scale(test.data,center=attr(ctrain.data.s2, "scaled:center"),scale=FALSE)

set.seed(1)
hdda.out.s2 <- hdda(ctrain.data.s2, train.target.s2, model="AKJBKQKD", d_select = "Cattell", threshold = 0.05)

#train error
pred.train.s2<-predict(hdda.out.s2, ctrain.data.s2, train.target.s2)
tab<-table(train.target.s2,pred.train.s2$class)
hdda.train.error.s2 <- 1-sum(diag(tab))/sum(tab)
print(hdda.train.error.s2)

#test error
pred.test.s2<-predict(hdda.out.s2, ctest.data.s2, test.target)
tab<-table(test.target,pred.test.s2$class)
hdda.test.error.s2 <- 1-sum(diag(tab))/sum(tab)
print(hdda.test.error.s2)

#overview table
row_lda <- c(lda.train.error.s1, lda.test.error.s1, lda.train.error.s2, lda.test.error.s2)
row_qda <- c(qda.train.error.s1, qda.test.error.s1, qda.train.error.s2, qda.test.error.s2)
row_knn <- c(knn.train.error.s1, knn.test.error.s1, knn.train.error.s2, knn.test.error.s2)
row_mlr <- c(mlr.train.error.s1, mlr.test.error.s1, mlr.train.error.s2, mlr.test.error.s2)
row_mlr_sq <- c(mlr.sq.train.error.s1, mlr.sq.test.error.s1, mlr.sq.train.error.s2, mlr.sq.test.error.s2)
row_gb <- c(gb.train.error.s1, gb.test.error.s1, gb.train.error.s2, gb.test.error.s2)
row_hdda <- c(hdda.train.error.s1, hdda.test.error.s1, hdda.train.error.s2, hdda.test.error.s2)

result1 <- rbind(row_lda, row_qda, row_knn, row_mlr, row_mlr_sq, row_gb, row_hdda)
colnames(result1) <- c("S1 Train", "S1 Test", "S2 Train", "S2 Test")
rownames(result1) <- c("LDA", "QDA", "KNN", "Multinom PC", 
                       "Multinom PC+PC^2", "Gradient Boosting", "HDDA")
print(round(result1,3))

# ------------------------------------------------------------------------------
# Step 3: Visualization of Results
# ------------------------------------------------------------------------------

df_plot <- as.data.frame(result1) %>%
  rownames_to_column(var = "Method") %>%
  pivot_longer(cols = -Method, names_to = "Scenario", values_to = "Error_Rate")


df_plot$Method <- factor(df_plot$Method, levels = rownames(result1))


p <- ggplot(df_plot, aes(x = Method, y = Error_Rate, group = Scenario, color = Scenario)) +
  
  geom_line(size = 1) +
  geom_point(size = 3) +
  
  scale_color_manual(values = c("S1 Train" = "#E41A1C", "S1 Test" = "#FF7F00", 
                                "S2 Train" = "#377EB8", "S2 Test" = "#4DAF4A")) +
  
  theme_minimal() +
  labs(title = "Benchmark of Classification Methods",
       subtitle = "Comparison of Error Rates across Scenarios (Lower is Better)",
       y = "Error Rate",
       x = "Classification Algorithm",
       caption = "Data Source: EMNIST Subset") +
  
  theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10, face = "bold"),
        legend.position = "top",
        legend.title = element_blank())

print(p)
ggsave("output/method_comparison.png", plot = p, width = 10, height = 6, dpi = 300)






