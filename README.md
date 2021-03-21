# Trivago RecSys Challenge 2019

The task is to predict whether a user will select (i.e. click on a selected) an item based on his/her online behavior during a session.

In order to predict the target value, we should transform the original data as follows.

We group the rows from the original dataset by user id, on top of that we compute:

1.  The total duration of the user by getting the fist and the last timestamps
2.  The steps that the user proceeded in order to click-out 
3.  The device that the user used
4.  The city name of the search context
5.  The country platform that was used for the search, e.g. trivago.de (DE) or trivago.com (US)
6.  The platform of the search context
7.  The actions that the user proceeded before click-out
8.  The sum of the filters that user user within a session
9.  The median of the hotel's prices
10. The sum of the impression when the user click-out
11. The sum of all the facilities that hotels are offering 

The generated csv from generateCsv.py (see on data folder export_dataframe.csv) consist of the following features:

1. durationOfSession
2. steps
3. device
3. city
3. country
4. platform
4. current_filters
5. impressions
5. hotel Facilities
6. interaction item image
7. search for poi
8. filter selection
9. interaction item info
10. search for destination
11. interaction item rating
12. change of sort order
13. interaction item deals
14. search for item
15. target

Gradient Descent
Correlation matrix
Variance Inflation Factor (VIF) is fo regression
https://medium.com/analytics-vidhya/what-is-multicollinearity-and-how-to-remove-it-413c419de2f
TODO: # Descriptive statistics for each column
features.describe()
feature_importances_
#TODO:  logistic regression, random forest regressor

## Data Description

## Data Preprocessing

### Detect Imbalanced Classes

In generated dataset the target values are 4985 rows for class:0 (not clickout) and 53544 rows for class:1 (clickout).

![alt text](https://raw.githubusercontent.com/SteveMyrsinias/Trivago/main/images/generatedCsvTargetDist.png?token=ACYAUFLN7FLEYZUPBVRJNFDAK4A7A)

As we can obser the dataset is imbalance, therefor we should delete some values from the majority class in order to balanche the data set

![alt text](https://raw.githubusercontent.com/SteveMyrsinias/Trivago/main/images/generatedCsvTargetDist1.png?token=ACYAUFNWWOLRFAHHWZIVJTLAK4A7G)

The final balanche dataset consist of 9970 observers.

TODO Corellation Matrix

### Detect Missing Data

### One Hot encoding
In order to convert the categorical variables as binary vectors we use the one hot encoding.
TODO: add the categorical values

### Splitting the data-set into Training and Test Set
This ensures that the random numbers are generated in the same order we use the random_state.

### Feature selection
https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e

#### Univariate Selection
Statistical tests can be used to select those features that have the strongest relationship with the output variable.

The scikit-learn library provides the SelectKBest class that can be used with a suite of different statistical tests to select a specific number of features.

#### Feature Importance
We use SelectKBest in order to select those features that they have the strongest relationship with the output variable.

### Feature Scaling

## Data Visualization

### 1.Gaussian Mixture

| Evaluation    | -           
| ------------- |:-------------:
| Accuracy      | - 
| Recall        | -     
| Precesion     | -     
| F-measure     | -      

Confusion Matrix:

TODO: ROC image 

### 2.Logistic Regression

| Evaluation    | -           
| ------------- |:-------------:
| Accuracy      | - 
| Recall        | -     
| Precesion     | -     
| F-measure     | -      

Confusion Matrix:

TODO: ROC image 

### 3.Decision Tree

| Evaluation    | -           
| ------------- |:-------------:
| Accuracy      | - 
| Recall        | -     
| Precesion     | -     
| F-measure     | -      

Confusion Matrix:

TODO: ROC image 

### 5.KNeighbors

| Evaluation    | -           
| ------------- |:-------------:
| Accuracy      | - 
| Recall        | -     
| Precesion     | -     
| F-measure     | -      

Confusion Matrix:

TODO: ROC image 

### 5.Random Forest

| Evaluation    | -           
| ------------- |:-------------:
| Accuracy      | - 
| Recall        | -     
| Precesion     | -     
| F-measure     | -      

Confusion Matrix:

TODO: ROC image 

### 6.Support Vector Machine

| Evaluation    | -           
| ------------- |:-------------:
| Accuracy      | - 
| Recall        | -     
| Precesion     | -     
| F-measure     | -      

Confusion Matrix:

TODO: ROC image 

### 7.MLPClassifier

| Evaluation    | -           
| ------------- |:-------------:
| Accuracy      | - 
| Recall        | -     
| Precesion     | -     
| F-measure     | -      

Confusion Matrix:

TODO: ROC image 
