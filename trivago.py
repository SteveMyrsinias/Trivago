########################################################################## START ##########################################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import svm,tree
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from functions import *
import time
import os
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.width', None)
elapsed_time = {"Gaussian Naive Bayes": [],"Logistic Regression": [] ,"KNeighbors": [],"Random Forest": [],"Decision Tree": [],"Support Vector Machine": [] ,"MLPClassifier": []} # Copute the computational time of every algorith
missing_values = ["n/a", "na", "--", "?"] # pandas only detect NaN, NA,  n/a and values and empty shell
my_path = os.path.abspath(os.path.dirname(__file__))
generated_df=pd.read_csv(r''+my_path+'\\data\\export_dataframe.csv',  sep=',', na_values=missing_values)

print('initial shape: ',generated_df.shape) # (58529, 16)
print('Description\n ',generated_df.describe()) # (58529, 16)
print('Missing values: ', generated_df.isnull().values.sum())

#### Start - Check for imbalanced dataset ####
print(generated_df.groupby(['target']).size()) # print the sum of every class,  0:4985, 1:53544

sns.countplot(data=generated_df,x=generated_df['target'])
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
sns.countplot(data=normalized_df,x=normalized_df['target'])
plt.title('Display the distribution of taget class')
plt.show()

print("generated_df", generated_df.shape) # (58529, 16)
print("normalized_df", normalized_df.shape)  # (9970, 16)
#### Stop - Check for imbalanced dataset ####

# Plot boxplot
displayBoxPlots(normalized_df, ['steps','current_filters','impressions','priceMean'])
displayUniqueBoxPlot(normalized_df, ['durationOfSession','hotel Facilities'])

# Plot Correlation HeatMap
correlationMatrix(normalized_df)

targetColumns = generated_df['target']
y = normalized_df.iloc[:, -1].values # get the target column
y = y.astype('int')
normalized_df.drop('target', axis=1, inplace=True) # drop the target column from data base

featureColumns = generated_df.columns
# Convert categorical variable into dummy
normalized_df = pd.get_dummies(normalized_df, columns=['device'],   prefix=['device_Type_is']     )
normalized_df = pd.get_dummies(normalized_df, columns=['city'],     prefix=['city_Type_is']       )
normalized_df = pd.get_dummies(normalized_df, columns=['country'],  prefix=['country_Type_is']    )
normalized_df = pd.get_dummies(normalized_df, columns=['platform'], prefix=['platform_Type_is']   )

print('country_Type_is: ', len(list(normalized_df.filter(regex='country_Type_is'))))
print('city_Type_is: ', len(list(normalized_df.filter(regex='city_Type_is'))))
print('platform_Type_is: ', len(list(normalized_df.filter(regex='platform_Type_is'))))

print('After Converting categorical variable into dummy/indicator variables: ', normalized_df.shape) 

# apply SelectKBest class to extract top 40 best features
excractFeatureImportance(normalized_df.iloc[:, :],y)

# Drop these features were does'n have any importance for the prediction
normalized_df = normalized_df[normalized_df.columns.drop(list(normalized_df.filter(regex='country_Type_is')))]
normalized_df = normalized_df[normalized_df.columns.drop(list(normalized_df.filter(regex='city_Type_is')))]
normalized_df = normalized_df[normalized_df.columns.drop(list(normalized_df.filter(regex='platform_Type_is')))]

print('After deleting columns : ', normalized_df.shape) 

X_train, X_test, y_train, y_test = train_test_split(normalized_df.values,y,test_size=0.3,random_state=109)


############################################################ Scale Data ############################################################################################

print('Min Before Scaling : ', np.min(X_train))
print('Max Before Scaling : ', np.max(X_train))

scaler = StandardScaler()
scaler.fit(X_train) # Fit on training set only.
X_train = scaler.transform(X_train) # Apply transform to both the training set and the test set.
X_test = scaler.transform(X_test)

print('Min After Scaling : ', np.min(X_train))
print('Max After Scaling : ', np.max(X_train))

########################################################################## PCA ##########################################################################

pca = PCA(.90)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

############################################################ Gaussian Naive Bayes ############################################################################################
modelName = 'Gaussian Naive Bayes'

# Grid Search
# param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
# gaussian = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid, cv=5,verbose=1, scoring='accuracy') 
# getModelsBestParameters(gaussian, modelName) # GaussianNB(var_smoothing=0.01)

# Create Model
start_gaussian = time.time()
gaussian = GaussianNB(var_smoothing=0.01)
gaussian.fit(X_train, y_train)
y_pred = gaussian.predict(X_test)
end_gaussian = time.time()

# Print Metrics
composeMetrics(y_test, y_pred, modelName)

# Print Trade off
measureTradeOffAlterNative(normalized_df.values,y,gaussian,10,modelName)
measureTradeOff(normalized_df.values,y,gaussian,10,modelName)

############################################################ Logistic Regression ############################################################################################
modelName = 'Logistic Regression'

start_logisticRegression = time.time()

# Grid Search
# clf = LogisticRegression()
# param_grid = {'C': [0.01, 0.1, 1, 2, 10, 100], 'penalty': ['l1', 'l2']}
# logreg = GridSearchCV(clf, param_grid = param_grid, cv = 5, verbose=True, n_jobs=-1)
# getModelsBestParameters(logreg, 'Logistic Regression') # LogisticRegression(C=100)

# Create Model
logreg = LogisticRegression(C=100)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
end_logisticRegression = time.time()

# predict_proba
# https://stackoverflow.com/questions/61184906/difference-between-predict-vs-predict-proba-in-scikit-learn

# Print Metrics
composeMetrics(y_test, y_pred, modelName)

# Print Trade off
measureTradeOffAlterNative(normalized_df.values,y,logreg,10,modelName)
measureTradeOff(normalized_df.values,y,logreg,10,modelName)

############################################################ Decision Tree ############################################################################################
modelName = 'Decision Tree'

# Grid Search
# param_grid = { 'max_leaf_nodes': list(range(2, 20)), 'min_samples_split': [2, 3],'max_depth': np.arange(3, 6)} #prone to overfitting
# decisionTree = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
# getModelsBestParameters(decisionTree, 'Decision Tree') # DecisionTreeClassifier(max_depth=5, max_leaf_nodes=18, min_samples_split=3)

# Create Model
start_decisionTree = time.time()
decisionTree = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=18, min_samples_split=3)
decisionTree.fit(X_train,y_train)
y_pred = decisionTree.predict(X_test)
end_decisionTree = time.time()

# Print Metrics
composeMetrics(y_test, y_pred, modelName)

# Print Trade off
measureTradeOffAlterNative(normalized_df.values,y,decisionTree,10,modelName)
measureTradeOff(normalized_df.values,y,decisionTree,10,modelName)

############################################################ KNeighbors ######################################################################################
modelName = 'KNeighbors'

# Grid Search
# grid_params = {'n_neighbors': [3,5,11,19],'weights': ['uniform','distance'],'metric': ['euclidean', 'manhattan']}
# kneighbors = GridSearchCV(KNeighborsClassifier(), grid_params, verbose=1, cv=5, n_jobs=1)
# getModelsBestParameters(kneighbors, 'KNeighbors Classifier') # KNeighborsClassifier(metric='euclidean', weights='distance')

# Create Model
start_KNeighbors = time.time()
kneighbors = KNeighborsClassifier(metric='euclidean', weights='distance')
kneighbors.fit(X_train,y_train)
y_pred = kneighbors.predict(X_test)
end_KNeighbors = time.time()

# Print Metrics
composeMetrics(y_test, y_pred, modelName)

# Print Trade off
measureTradeOffAlterNative(normalized_df.values,y,kneighbors,10,modelName)
measureTradeOff(normalized_df.values,y,kneighbors,10,modelName)

############################################################ Random Forest #################################################################################
modelName = 'Random Forest'

# Grid Search
# param_grid = { 'n_estimators': [200, 700],'max_features': ['auto', 'sqrt', 'log2']}
# rf = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
# getModelsBestParameters(rf, 'Random Forest') # RandomForestClassifier(max_features='log2', n_estimators=200, n_jobs=-1, oob_score=True)

# Create Model
start_RandomForest = time.time()
rf = RandomForestClassifier(max_features='log2', n_estimators=200, n_jobs=-1, oob_score=True)
rf.fit(X_train,y_train)
y_pred = rf.predict(X_test)
end_RandomForest = time.time()

# Print Metrics
composeMetrics(y_test, y_pred, modelName)

# Print Trade off
measureTradeOffAlterNative(normalized_df.values,y,rf,10,modelName)
measureTradeOff(normalized_df.values,y,rf,10,modelName)

################################################################## MLPClassifier ###################################################################
modelName = 'MLPClassifier'

# Grid Search
# mlp = MLPClassifier(max_iter=100)
# parameter_space = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],'activation': ['tanh', 'relu'],'solver': ['sgd', 'adam'],'alpha': [0.0001, 0.05],'learning_rate': ['constant','adaptive'],}
# clfANN = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
# getModelsBestParameters(clfANN, 'MLPClassifier') # MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(50, 50, 50),learning_rate='adaptive', max_iter=100)

# Create Model
start_MPLClassifier = time.time()
clfANN =MLPClassifier(activation='tanh', alpha=0.05, hidden_layer_sizes=(2, 2, 2),learning_rate='adaptive', max_iter=100)
clfANN.fit(X_train, y_train)                         
y_pred=clfANN.predict(X_test)
end_MPLClassifier = time.time()

# Print Metrics
composeMetrics(y_test, y_pred, modelName)

# Print Trade off
measureTradeOffAlterNative(normalized_df.values,y,clfANN,10,modelName)
measureTradeOff(normalized_df.values,y,clfANN,10,modelName)

############################################################ Support Vector Machine #########################################################################
modelName = 'Support Vector Machine'

# Grid Search
# parameters = {'kernel':('linear', 'rbf'), 'C':(1,0.25,0.5,0.75),'gamma': (1,2,3,'auto'),'decision_function_shape':('ovo','ovr'),'shrinking':(True,False)}
# clf = GridSearchCV(svm.SVC(), parameters,cv=5)
# getModelsBestParameters(clf, 'Support Vector Machine SVM') #SVC(C=1, decision_function_shape='ovo', gamma=1, kernel='linear')

# Create Model
start_SupportVectorMachine = time.time()
clf = svm.SVC(C=1, decision_function_shape='ovo', gamma=1, kernel='rbf')
#clf = svm.SVC(C=1, decision_function_shape='ovo', gamma=1, kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end_SupportVectorMachine = time.time()

# Print Metrics
composeMetrics(y_test, y_pred, modelName)

# Print Trade off
# Comment below due to freeze of code
# measureTradeOffAlterNative(normalized_df.values,y,clf,10,modelName)
# measureTradeOff(normalized_df.values,y,clf,10,modelName)

############################################ Elapsed time per model ###################################################################

elapsed_time["Gaussian Naive Bayes"].append(round(end_gaussian-start_gaussian,2))
elapsed_time["Logistic Regression"].append(round(end_logisticRegression-start_logisticRegression,2))
elapsed_time["KNeighbors"].append(round(end_KNeighbors-start_KNeighbors,2))
elapsed_time["Random Forest"].append(round(end_RandomForest-start_RandomForest,2))
elapsed_time["Decision Tree"].append(round(end_decisionTree-start_decisionTree,2))
elapsed_time["Support Vector Machine"].append(round(end_SupportVectorMachine-start_SupportVectorMachine,2))
elapsed_time["MLPClassifier"].append(round(end_MPLClassifier-start_MPLClassifier,2))

for x in elapsed_time:
  print('Computation Time of ' + x + ':', elapsed_time[x])
########################################################################## END ##########################################################################