# Trivago RecSys Challenge 2019

The task is to predict whether a user will select (i.e. click on a selected) an item based on his/her online behavior during a session.

In order to predict the target value, we should transform the original data as follows.

We group the rows from the original dataset by user id, on top of that we compute

1. The total duration of the user by getting the fist and the last timestamps
2. The steps that the user need in order to click-out 
3. The device that was used for the search
3. The name of the current city of the search context
4. The country platform that was used for the search, e.g. trivago.de (DE) or trivago.com (US)

The generated csv (see on data folder export_dataframe.csv) consist of the following features:

1. durationOfSession
2. steps
3. device
3. city
4. country
5. platform
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

## Data Description

At generated dataset the target values are 4985 rows for class:0 (not clickout) and 53544 rows for class:1 (clickout).
As we can obser the dataset is imbalance, there for we should delete some values from the majority class.

TODO: add images generatedCsvTargetDist.png generatedCsvTargetDist1.png

## Data Visualization

## Dataset Cleaning
Since the target is to classify if the user is going to click-out we can omit from action_type column the search for destination, filter selection 
