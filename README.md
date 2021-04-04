# Trivago RecSys Challenge 2019

The task is to predict whether a user will select (i.e. click on a selected) an item based on his/her online behavior during a session.

In order to predict the target value, we should transform the original data as follows.

We group the rows from the original dataset by user id, on top of that we compute:

1.  The total duration of the user's session by subtract the last timestamp from the fist timestamp.
2.  The total steps that the user proceeded. 
3.  The device that the user used for the search.
4.  The current city of the search context.
5.  The current country of the search context.
6.  The country platform that was used for the search, e.g. trivago.de (DE) or trivago.com (US)
7.  The total count of filters tha user proceed in a session.
8.  The sum of the impression was shown when the user click-out.
9.  The median of the hotel's prices that displayed in user.
10. The sum of all the facilities that hotels are offering .
11. Create a column for each type of action and count the total action of them.
12. Create a target column to idicate if the user click out or not.

The generated csv from generateCsv.py (see on data folder export_dataframe.csv) consist of the following features:

1. durationOfSession
2. steps
3. device
4. city
5. country
6. platform
7. current_filters
8. impressions
9. priceMean
10. hotel Facilities
11. interaction item image
12. search for poi
13. filter selection
14. interaction item info
15. search for destination
16. interaction item rating
17. change of sort order
18. interaction item deals
19. search for item
20. target

### Data Description

Initial dataset: (1048575, 12)
Generated dataset: (58529, 20)
Print missing values:  0

### Correlation HeatMap

![](images/correlationHeatMap.png)

### Data Preprocessing


#### Detect Imbalanced Classes

In generated dataset the target values are 4985 rows for class:0 (not clickout) and 53544 rows for class:1 (clickout).

![](images/initialDistOfTargetClass.png)

As we can obser that the dataset is imbalance, therefor we should delete some values from the majority class in order to balance the data set.

![](images/generatedDistOfTargetClass.png)

The final balanced dataset consist of 9970 observers.



### Detect Outliers

![](images/boxPlot1.png)
![](images/boxPlot2.png)
![](images/boxPlot3.png)

### One Hot encoding
In order to convert the categorical variables as binary vectors we use the one hot encoding.
TODO: add the categorical values

### Splitting the data-set into Training and Test Set
This ensures that the random numbers are generated in the same order we use the random_state.

#### Feature Importance
We use SelectKBest in order to select those features that they have the strongest relationship with the output variable.

![](images/featureImportances.png)

### Feature Scaling

## Data Visualization

### 1.Gaussian Mixture

![](images/gaussianNaiveBayesConfusionMatrix.png)
![](images/GaussianNaiveBayesRocCurve.png)
![](images/gaussianNaiveBayesTradeOff1.png)
![](images/gaussianNaiveBayesTradeOff2.png)

| Evaluation    | Gaussian Mixture          
| ------------- |:-------------:
| Accuracy      | 0.98 %
| Recall        | 0.98 %     
| Precesion     | 0.99 %     
| F-measure     | 0.98 %      

### 2.Logistic Regression

![](images/logisticRegressionConfusionMatrix.png)
![](images/logisticRegressionRocCurve.png)
![](images/logisticRegressionTradeOff1.png)
![](images/logisticRegressionTradeOff2.png)

| Evaluation    | Logistic Regression           
| ------------- |:-------------:
| Accuracy      | 1.0 % 
| Recall        | 1.0 %     
| Precesion     | 1.0 %     
| F-measure     | 1.0 %      
| Macro Precision, recall, f1-score | 1.0, 1.0, 1.0
| Micro Precision, recall, f1-score | 1.0, 1.0, 1.0

### 3.Decision Tree

![](images/decisionTreeConfusionMatrix.png)
![](images/decisionTreeRocCurve.png)
![](images/decesionTreeTradeOff1.png)
![](images/decesionTreeTradeOff2.png)

| Evaluation    | Decision Tree           
| ------------- |:-------------:
| Accuracy      | 0.99 % 
| Recall        | 0.99 %     
| Precesion     | 1.0 %     
| F-measure     | 0.99 %      
| Macro Precision, recall, f1-score | 0.9932786056746217, 0.9933514195357824, 0.9933110113589807
| Micro Precision, recall, f1-score | 0.9933132731527917, 0.9933132731527917, 0.9933132731527917

### 5.KNeighbors

![](images/kneighborsConfusionMatrix.png)
![](images/kneighborsRocCurve.png)
![](images/kneighborsTradeOff1.png)
![](images/kneighborsTradeOff2.png)

| Evaluation    | KNeighbors           
| ------------- |:-------------:
| Accuracy      | 0.99 % 
| Recall        | 0.99 %     
| Precesion     | 1.0 %     
| F-measure     | 0.99 %      
| Macro Precision, recall, f1-score | 0.987608360891175, 0.9878085874422411, 0.987627956634628
| Micro Precision, recall, f1-score | 0.9876295553326646, 0.9876295553326646, 0.9876295553326646

### 5.Random Forest

![](images/randomForestConfusionMatrix.png)
![](images/randomForestRocCurve.png)
![](images/randomForestTradeOff1.png)
![](images/randomForestTradeOff2.png)

| Evaluation    | Random Forest          
| ------------- |:-------------:
| Accuracy      | 0.99 % 
| Recall        | 0.99 %     
| Precesion     | 1.0 %     
| F-measure     | 0.99 %      
| Macro Precision, recall, f1-score | 0.9946037744286929, 0.994716607249093, 0.994649062780931
| Micro Precision, recall, f1-score | 0.9946506185222334, 0.9946506185222334, 0.9946506185222334

### 6.Support Vector Machine

![](images/supportVectorMachineConfusionMatrix.png)
![](images/supportVectorMachineRocCurve.png)


| Evaluation    | Support Vector Machine          
| ------------- |:-------------:
| Accuracy      | 1.0 % 
| Recall        | 1.0 %     
| Precesion     | 1.0 %     
| F-measure     | 1.0 %     
| Macro Precision, recall, f1-score | 1.0, 1.0, 1.0
| Micro Precision, recall, f1-score | 1.0, 1.0, 1.0   

### 7.MLPClassifier

![](images/mlpClassifierConfusionMatrix.png)
![](images/mlpClassifierRocCurve.png)
![](images/mlpClassifierTradeOff1.png)
![](images/mlpClassifierTradeOff2.png)

| Evaluation    | MLPClassifier          
| ------------- |:-------------:
| Accuracy      | 1.0 % 
| Recall        | 1.0 %     
| Precesion     | 1.0 %     
| F-measure     | 1.0 %      
| Macro Precision, recall, f1-score | 0.9986494211229732, 0.998675741079536, 0.9986621340549728
| Micro Precision, recall, f1-score | 0.9986626546305584, 0.9986626546305584, 0.9986626546305584  

### Computation Time

| Model | Time (in seconds) |
| --- | --- |
| Gaussian Naive Bayes | 0.3 |
| Logistic Regression: | 0.06 |
| KNeighbors: | 0.13 |
| Random Forest | 0.78 |
| Decision Tree | 0.06 |
| Support Vector Machine | 0.24 |
| MLPClassifier: | 1.42 |

### Feature Work 
Analysis mote the current_filters column