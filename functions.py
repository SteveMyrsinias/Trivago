########################################################################## START ##########################################################################
from sklearn.metrics import mean_absolute_error, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from seaborn import countplot
import seaborn as sns
import numpy as np
import pandas as pd

def plotConfusionMatrix(y_test, y_pred , model):
   cm = confusion_matrix(y_test, y_pred)
   cm_df = pd.DataFrame(cm, index = ['Clickout','Not Clickout'], columns = ['Clickout','Not Clickout'])
   plt.figure(figsize=(5.5,4))
   sns.heatmap(cm_df, annot=True)
   plt.title(model + ' Confusion Matrix')
   plt.ylabel('True label')
   plt.xlabel('Predicted label')
   plt.show()

def displayUniqueBoxPlot(df, columns):
   for col in columns:
      df.boxplot(column =[col], grid = True)
      plt.show()

def displayBoxPlots(df, columsToBeDisplayed):
   columnIndexes = []
   for col in columsToBeDisplayed:
      columnIndexes.append(df.columns.get_loc(col))
   centroids = pd.DataFrame(df.iloc[:, columnIndexes], columns=columsToBeDisplayed)
   sns.boxplot(data=centroids)
   plt.show()

def printMetrics(y_test, y_pred, algoName):
   print(algoName + ' Accuracy: ',           round(accuracy_score(y_test, y_pred), 2), '%')
   print(algoName + ' Recall: ',             round(recall_score(y_test, y_pred), 2), '%')
   print(algoName + ' Precesion: ',          round(precision_score(y_test, y_pred), 2), '%')
   print(algoName + ' F-measure: ',          round(f1_score(y_test, y_pred), 2), '%')
   print(algoName + ' Confusion Matrix: \n', confusion_matrix(y_test, y_pred))

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
   plt.title('Plot Correlation HeatMap')
   plt.show()

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
   feat_importances.nlargest(40).plot(kind='barh')
   plt.title('Feature Importances')
   plt.show()

def getModelsBestParameters(model, algoName):
   print(algoName + ' Best Parameters : ', model.best_estimator_)

def printMicroMacroMetrics(y_test, y_pred, modelName):
   print (modelName + ' : Macro Precision, recall, f1-score', precision_recall_fscore_support(y_test, y_pred, average='macro'))
   print (modelName + ' : Micro Precision, recall, f1-score', precision_recall_fscore_support(y_test, y_pred, average='micro'))

def composeMetrics(y_test, y_pred, modelName):
   printMicroMacroMetrics(y_test, y_pred, modelName)
   printMetrics(y_test, y_pred, modelName)
   pltRocCurve(y_test, y_pred, modelName)
   plotConfusionMatrix(y_test, y_pred, modelName)

def measureTradeOff(X,y,model,n_splits,modelName):
   y = np.array(y)
   kf = KFold(n_splits=n_splits)
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
   plt.title(modelName + ': Training error across folds')
   plt.tight_layout()
   plt.subplot(1,2,2)
   plt.plot(range(1, kf.get_n_splits() + 1), np.array(list_testing_error).ravel(), 'o-')
   plt.xlabel('Number of fold')
   plt.ylabel('Testing error')
   plt.title(modelName + ': Testing error across folds')
   plt.tight_layout()
   plt.show()


def measureTradeOffAlterNative(X,y,model,n_splits,modelName):
   y = np.array(y)
   kfold = KFold(n_splits=n_splits)
   train_scores=[]
   test_scores=[]
   for train, test in kfold.split(X): 
      X_train, X_test = X[train], X[test]
      y_train, y_test = y[train], y[test]
      model.fit(X_train,y_train)
      train_score=model.score(X_train,y_train)
      test_score=model.score(X_test,y_test)
      train_scores.append(train_score)
      test_scores.append(test_score)

   plt.plot(train_scores, color='red',label='Training Accuracy')
   plt.plot(test_scores, color='blue',label='Testing Accuracy')
   plt.xlabel('K values')
   plt.ylabel('Accuracy Score')
   plt.title(modelName + ': Performace Under Varying K Values')  
   plt.legend()
   plt.show() 

########################################################################## END ##########################################################################