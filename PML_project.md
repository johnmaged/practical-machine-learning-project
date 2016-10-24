# Practical Machine Learning Course Project

##Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

```
library(caret)
library(png)
library(grid)
library(gbm)
```

##Cleaning and Preparing Data

```
setwd("E:\\PML3")
train_in <- read.csv('./pml-training.csv', header=T)
validation <- read.csv('./pml-testing.csv', header=T)
```

####Data Partitioning
Now, I will split the training data `pml-training.csv` into training and testing partitions and use the `pml-testing.csv` as a validation sample. 

Then I will use cross validation within the training partition to improve the model fit and then do an out-of-sample test with the testing partition.

```
set.seed(127)
training_sample <- createDataPartition(y=train_in$classe, p=0.7, list=FALSE)
training <- train_in[training_sample, ]
testing <- train_in[-training_sample, ]
```

####Identification on Non-Zero Data

In order to predict classes in the validation sample, I will need to use features that are non-zero in the validation data set. Typically, I'd stay away from the even looking at the validation data set so I'm not influenced by the contents in model fitting. However, since this is not a time series analysis, I feel that looking at the validation sample for non-zero data columns is not of major concern for finding a predictive model that fits well out of sample.

```
all_zero_colnames <- sapply(names(validation), function(x) all(is.na(validation[,x])==TRUE))
nznames <- names(all_zero_colnames)[all_zero_colnames==FALSE]
nznames <- nznames[-(1:7)]
nznames <- nznames[1:(length(nznames)-1)]
```

The models will be fit using the following data columns:

```
print(sort(nznames))
```

##Model building
In this project, I'll use `3 differnt model algorithms` and then look to see which one provides the best accuracty. 

*The three model types I'm going to test are:*
1. Decision trees with CART (*rpart*)
2. Stochastic gradient boosting trees (*gbm*)
3. Random forest decision trees (*rf*)

The code to run fit these models is:
```
fitControl <- trainControl(method='cv', number = 3)
```

```{r, eval=FALSE}
model_cart <- train(
  classe ~ ., 
  data=training[, c('classe', nznames)],
  trControl=fitControl,
  method='rpart'
)
save(model_cart, file='./ModelFitCART.RData')
model_gbm <- train(
  classe ~ ., 
  data=training[, c('classe', nznames)],
  trControl=fitControl,
  method='gbm'
)
save(model_gbm, file='./ModelFitGBM.RData')
model_rf <- train(
  classe ~ ., 
  data=training[, c('classe', nznames)],
  trControl=fitControl,
  method='rf',
  ntree=100
)
save(model_rf, file='./ModelFitRF.RData')
```


####Cross Validation

Cross validation is done for each model with `K = 3`. This is set in the above code chunk using the fitControl object as defined below:

```
fitControl <- trainControl(method='cv', number = 3)
```

####Model Assessment

```
load('./ModelFitCART.RData')
load('./ModelFitGBM.RData')
load('./ModelFitRF.RData')
```

**Working with Decision trees with CART (rpart)**

```
predCART <- predict(model_cart, newdata=testing)
cmCART <- confusionMatrix(predCART, testing$classe)
plot(predCART)
```
(fig01.png)

**Working with Stochastic gradient boosting trees (gbm)**

```
predGBM <- predict(model_gbm, newdata=testing)
cmGBM <- confusionMatrix(predGBM, testing$classe)
plot(predGBM)
```
(fig02.png)

**Working with Random forest decision trees (rf)**

```
predRF <- predict(model_rf, newdata=testing)
cmRF <- confusionMatrix(predRF, testing$classe)
plot(predRF)
```
(fig03.png)

**Calculating Accuracy.**

```
AccuracyResults <- data.frame(
  Model = c('CART', 'GBM', 'RF'),
  Accuracy = rbind(cmCART$overall[1], cmGBM$overall[1], cmRF$overall[1])
)
print(AccuracyResults)
```

Based on an assessment of these 3 model fits and out-of-sample results, it looks like both gradient boosting and random forests outperform the CART model, with random forests being slightly more accurate. The confusion matrix for the random forest model is shown below.

```
print(cmRF$table)
```

The next step is to create an ensemble model of these three model results, however, given the high accuracy of the random forest model, I don't believe this process is necessary here. I'll accept the random forest model as the champion and move on to prediction in the validation sample.

```
champion_model <- model_rf
```

```
imp <- varImp(champion_model)
imp$importance$Overall <- sort(imp$importance$Overall, decreasing=TRUE)
featureDF <- data.frame(
  FeatureName=row.names(imp$importance),
  Importance=imp$importance$Overall
)
```

**The champion model includes the following 5 features as the most important ones**. 

```
print(featureDF[1:5,])
```

**Here is below a feature plot that show how these features are related to one another and how clusters of exercise class begin to appear using these 5 features.**

```
featurePlot(x=training[, featureDF$FeatureName[1:5]], y=training$classe, plot='pairs')
```
(fig04.png)

##Prediction

Now, I'll use the validation data sample `pml-testing.csv` to predict a class for each of the 20 observations based on the other information we know about these observations contained in the validation sample.

```
predValidation <- predict(champion_model, newdata=validation)
ValidationPredictionResults <- data.frame(
  problem_id=validation$problem_id,
  predicted=predValidation
)
print(ValidationPredictionResults)
```

#Conclusion

Based on all of the above, I am able to fit a reasonably sound model with a high degree of accuracy in predicting out of sample observations. 

The **random forest model with cross-validation** produces a surprisingly accurate model that is sufficient for predictive analytics.