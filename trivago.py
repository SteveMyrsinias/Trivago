from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, confusion_matrix, precision_recall_fscore_support, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score,KFold,validation_curve, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn import tree
import matplotlib.pyplot as plt
from seaborn import countplot
import seaborn as sns
import numpy as np
import pandas as pd
import os

def printMetrics(y_test, y_pred, algoName):
   print('\n')
   print(algoName + ' Accuracy: ',           round(accuracy_score(y_test, y_pred), 2), '%.')
   print(algoName + ' Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
   print(algoName + ' Recall: ',             round(recall_score(y_test, y_pred), 2), '%.')
   print(algoName + ' Precesion: ',          round(precision_score(y_test, y_pred), 2), '%.')
   print(algoName + ' F-measure: ',          round(f1_score(y_test, y_pred), 2), '%.')

def pltRocCurve(y_test, y_pred, algoName):
   # ROC-AUC curve
   # calculate the fpr and tpr for all thresholds of the classification
   fpr, tpr, threshold = roc_curve(y_test, y_pred)
   roc_auc = auc(fpr, tpr)

   # method I: plt
   plt.title('Receiver Operating Characteristic for ' + algoName)
   plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
   plt.legend(loc = 'lower right')
   plt.plot([0, 1], [0, 1],'r--')
   plt.xlim([0, 1])
   plt.ylim([0, 1])
   plt.ylabel('True Positive Rate')
   plt.xlabel('False Positive Rate')
   plt.show()

def correlationMatrix(inutDataFrame):
   corrMatrix = inutDataFrame.corr()
   sns.heatmap(corrMatrix, annot=True)
   plt.show()

def printMissingPercentageForEveryFeature(inutDataFrame):
   for col in inutDataFrame.columns:
      pct_missing = np.mean(inutDataFrame[col].isnull())
      print('{} - {}%'.format(col, round(pct_missing*100)))

def excractBestFeatures(X,y,bestFeaturesNum):
   # apply SelectKBest class to extract top 10 best features
   bestfeatures = SelectKBest(score_func=chi2, k=bestFeaturesNum)
   fit = bestfeatures.fit(X,y)
   dfscores = pd.DataFrame(fit.scores_)
   dfcolumns = pd.DataFrame(X.columns)

   # concat two dataframes for better visualization 
   featureScores = pd.concat([dfcolumns,dfscores],axis=1)
   featureScores.columns = ['Specs','Score']  # naming the dataframe columns
   print(featureScores.nlargest(bestFeaturesNum,'Score'))  # print 10 best features

def excractFeatureImportance(X,y):
   model = ExtraTreesClassifier()
   model.fit(X,y)
   #print(model.feature_importances_) # use inbuilt class feature_importances of tree based classifiers
   # plot graph of feature importances for better visualization
   feat_importances = pd.Series(model.feature_importances_, index=X.columns)
   feat_importances.nlargest(10).plot(kind='barh')
   plt.title('Feature Importances ')
   plt.show()

def getModelsBestParameters(model, algoName):
   print(algoName + ' Best Parameters : ', model.best_estimator_)

def measureTradeOff(X,y,model):
   y = np.array(y)
   kf = KFold(n_splits=10)
   list_training_error = []
   list_testing_error = []
   for train_index, test_index in kf.split(X):
       X_train, X_test = X[train_index], X[test_index]
       y_train, y_test = y[train_index], y[test_index]
       model.fit(X_train, y_train)
       y_train_data_pred = model.predict(X_train)
       y_test_data_pred = model.predict(X_test)
       fold_training_error = mean_absolute_error(y_train, y_train_data_pred) 
       fold_testing_error = mean_absolute_error(y_test, y_test_data_pred)
       list_training_error.append(fold_training_error)
       list_testing_error.append(fold_testing_error)

   plt.subplot(1,2,1)
   plt.plot(range(1, kf.get_n_splits() + 1), np.array(list_training_error).ravel(), 'o-')
   plt.xlabel('Number of fold')
   plt.ylabel('Training error')
   plt.title('Training error across folds')
   plt.tight_layout()
   plt.subplot(1,2,2)
   plt.plot(range(1, kf.get_n_splits() + 1), np.array(list_testing_error).ravel(), 'o-')
   plt.xlabel('Number of fold')
   plt.ylabel('Testing error')
   plt.title('Testing error across folds')
   plt.tight_layout()
   plt.show()

missing_values = ["n/a", "na", "--", "?"] # pandas only detect NaN, NA,  n/a and values and empty shell
my_path = os.path.abspath(os.path.dirname(__file__))
generated_df=pd.read_csv(r''+my_path+'\\data\\export_dataframe.csv', nrows=10000, sep=',', na_values=missing_values)

print(generated_df.shape) # (58529, 16)

#### Start - Check for imbalanced dataset ####
print(generated_df.groupby(['target']).size()) # print the sum of every class,  0:4985, 1:53544

countplot(data=generated_df,x=generated_df['target'])
plt.title('Display the distribution of taget class')
plt.show()

# Undersampling 
# Is the process where you randomly delete some of the observations 
# from the majority class in order to match the numbers with the minority class.

# Shuffle the Dataset.
shuffled_df = generated_df.sample(frac=1,random_state=4)
print("shuffled_df ", shuffled_df.shape)  # (58529, 16)

# Put all the 0 class (minority) in a separate dataset.
fraud_df = shuffled_df.loc[shuffled_df['target'] == 0]

#Randomly select 4985 observations from the 1 (majority class)
non_fraud_df = shuffled_df.loc[shuffled_df['target'] == 1].sample(n=4985,random_state=42)

# Concatenate both dataframes again
normalized_df = pd.concat([fraud_df, non_fraud_df])

# plot the dataset after the undersampling
countplot(data=normalized_df,x=normalized_df['target'])
plt.title('Display the distribution of taget class')
plt.show()

print("generated_df", generated_df.shape) # (58529, 16)
print("normalized_df", normalized_df.shape)  # (9970, 16)
#### Stop - Check for imbalanced dataset ####

# Plot Correlation HeatMap
correlationMatrix(normalized_df)

# print percentage of null values per feature
printMissingPercentageForEveryFeature(normalized_df)

targetColumns = generated_df['target']
y = normalized_df.iloc[:, -1].values # get the target column
y = y.astype('int')
normalized_df.drop('target', axis=1, inplace=True) # drop the target column from data base

featureColumns = generated_df.columns
# convert categorical variable into dummy
normalized_df = pd.get_dummies(normalized_df, columns=['device'],   prefix=['device_Type_is']     )
normalized_df = pd.get_dummies(normalized_df, columns=['city'],     prefix=['city_Type_is']       )
normalized_df = pd.get_dummies(normalized_df, columns=['country'],  prefix=['country_Type_is']    )
normalized_df = pd.get_dummies(normalized_df, columns=['platform'], prefix=['platform_Type_is']   )

# apply SelectKBest class to extract top 10 best features
excractFeatureImportance(normalized_df.iloc[:, :-1],y)

# Scale Data
scaled_features = normalized_df.copy()
columns = scaled_features[normalized_df.columns]
std_scale = StandardScaler().fit(columns.values)
X = std_scale.transform(columns.values)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=109)

############################################################ Gaussian ############################################################################################

nb_classifier = GaussianNB()
params_NB = {'var_smoothing': np.logspace(0,-9, num=100)}
gaussian = GridSearchCV(estimator=nb_classifier, 
                 param_grid=params_NB, 
                 cv=2,   # use any cross validation technique 
                 verbose=1, 
                 scoring='accuracy') 

#gnb = GaussianNB()
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)

getModelsBestParameters(gaussian, 'Gaussian Naive Bayes')
printMetrics(y_test, y_pred, 'Gaussian Naive Bayes')
pltRocCurve(y_test, y_pred, 'Gaussian Naive Bayes')
measureTradeOff(X,y,gaussian)

############################################################ Logistic Regression ############################################################################################

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

getModelsBestParameters(logreg, 'Logistic Regression')
printMetrics(y_test, y_pred, 'Logistic Regression')
pltRocCurve(y_test, y_pred, 'Logistic Regressions')

############################################################ Decision Tree ############################################################################################

param_grid = { 
    'max_leaf_nodes': list(range(2, 20)), 
    'min_samples_split': [2, 3],
    'max_depth': np.arange(3, 6)
}

decisionTree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=2)
decisionTree.fit(X_train,y_train)
y_pred = decisionTree.predict(X_test)

getModelsBestParameters(decisionTree, 'Decision Tree')
printMetrics(y_test, y_pred, 'Decision Tree')
pltRocCurve(y_test, y_pred, 'Decision Tree')

############################################################ KNeighbors #################################################################################################

kneighbors = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
kneighbors.fit(X_train,y_train)
y_pred = kneighbors.predict(X_test)

getModelsBestParameters(kneighbors, 'KNeighbors Classifier')
printMetrics(y_test, y_pred, 'KNeighbors Classifier')
pltRocCurve(y_test, y_pred, 'KNeighbors Classifier')

############################################################ Random Forest #############################################################################################

RSEED = 50
rf = RandomForestClassifier(n_estimators=100, 
                               random_state=RSEED, 
                               max_features = 'sqrt',
                               n_jobs=-1, verbose = 1)

rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)

###### Visualize all the trees ######
# fn=featureColumns
# cn=targetColumns
# fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
# for index in range(0, 5):
#     tree.plot_tree(rf.estimators_[index],
#                    feature_names = fn, 
#                    class_names=cn,
#                    filled = True,
#                    ax = axes[index])

#     axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
# fig.savefig('rf_5trees.png')
###### Visualize all the trees ######

print(rf.estimators_) # all the individual trees

getModelsBestParameters(rf, 'Random Forest')
printMetrics(y_test, y_pred, 'Random Forest')
pltRocCurve(y_test, y_pred, 'Random Forest')

############################################################ Support Vector Machine #####################################################################################

parameters = {
    'kernel':('linear', 'rbf'), 
    'C':(1,0.25,0.5,0.75),
    'gamma': (1,2,3,'auto'),
    'decision_function_shape':('ovo','ovr'),
    'shrinking':(True,False)
}

clf = GridSearchCV(svm.SVC(), parameters)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

getModelsBestParameters(clf, 'Support Vector Machine SVM')
printMetrics(y_test, y_pred, 'Support Vector Machine SVM')
pltRocCurve(y_test, y_pred, 'Support Vector Machine SVM')

########################################################################## MLPClassifier #############################################################################

clfANN = MLPClassifier(solver='sgd', activation='logistic',
                       batch_size=10,
                       hidden_layer_sizes=(2,2), random_state=1, max_iter=1000, verbose=True)

#train the classifiers
clfANN.fit(X_train, y_train)                         

#test the trained model on the test set
y_test_pred_ANN=clfANN.predict(X_test)

confMatrixTestANN=confusion_matrix(y_test, y_test_pred_ANN, labels=None)

getModelsBestParameters(clfANN, 'MLPClassifier')
print ('Conf matrix Neural Net')
print (confMatrixTestANN)

# Measures of performance: Precision, Recall, F1
print ('NearNeigh: Macro Precision, recall, f1-score')
print ( precision_recall_fscore_support(y_test, y_test_pred_ANN, average='macro'))
print ('NearNeigh: Micro Precision, recall, f1-score')
print (precision_recall_fscore_support(y_test, y_test_pred_ANN, average='micro'))
print ('\n')

y = np.array(y)
kf = KFold(n_splits=10)
list_training_error = []
list_testing_error = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = MLPClassifier(solver='sgd', activation='logistic',
                       batch_size=10,
                       hidden_layer_sizes=(5,5), random_state=1, max_iter=100, verbose=True)
    model.fit(X_train, y_train)
    y_train_data_pred = model.predict(X_train)
    y_test_data_pred = model.predict(X_test)
    fold_training_error = mean_absolute_error(y_train, y_train_data_pred) 
    fold_testing_error = mean_absolute_error(y_test, y_test_data_pred)
    list_training_error.append(fold_training_error)
    list_testing_error.append(fold_testing_error)

plt.subplot(1,2,1)
plt.plot(range(1, kf.get_n_splits() + 1), np.array(list_training_error).ravel(), 'o-')
plt.xlabel('number of fold')
plt.ylabel('training error')
plt.title('Training error across folds')
plt.tight_layout()
plt.subplot(1,2,2)
plt.plot(range(1, kf.get_n_splits() + 1), np.array(list_testing_error).ravel(), 'o-')
plt.xlabel('number of fold')
plt.ylabel('testing error')
plt.title('Testing error across folds')
plt.tight_layout()
plt.show()
