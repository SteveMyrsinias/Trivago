import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os

missing_values = ["n/a", "na", "--", "?"] # pandas only detect NaN, NA,  n/a and values and empty shell
my_path = os.path.abspath(os.path.dirname(__file__))
df_train=pd.read_csv(r''+my_path+'\\data\\train.csv', sep=',', nrows=1000,na_values=missing_values)
df_test=pd.read_csv(r''+my_path+'\\data\\test.csv', sep=',', nrows=1000,na_values=missing_values)
print(df_train.shape)

indexs = df_train[ (df_train['action_type']=='search for poi') | (df_train['action_type']=='filter selection') ].index
df_train.drop(indexs, inplace=True)
print(df_train.shape)

users = df_train['user_id'].nunique()
print('Users: ', users)

reference = df_train['reference'].nunique()
print('reference: ',reference)

sessions = df_train['session_id'].nunique()
print('Sessions: ',sessions)

actionType = df_train['action_type'].nunique()
print('Action Type: ',actionType)

myvar  = df_train.groupby(['action_type'])['action_type'].count()
print(myvar)
print(type(myvar))
# y = np.array([myvar[0], myvar[1], myvar[2], myvar[3], myvar[4], myvar[5], myvar[6], myvar[7] ])
pieCharLabeles = ['change of sort order','clickout item ','interaction item deals','interaction item image','interaction item info','interaction item rating','search for destination','search for item']
plt.pie(myvar.values, labels=pieCharLabeles)
plt.show()

print(df_train.groupby(df_train['user_id'],as_index=False).size())
s = df_train.groupby(df_train['user_id'],as_index=False).size()
print(type(s))
print('max ', s.max())
print(type(s.max()))
print('min ', s.min())
print('mean ', s.mean())
num_bins = 5
# plt.hist(s.max().values, num_bins, facecolor='blue', alpha=0.5)
# plt.show()

print(df_train.head(5))
df_train =  StandardScaler().fit(df_train)
print(df_train.head(5))

# X = df_train.iloc[:, 0:4]
# y = df_train.iloc[:, 4]



# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.5)

# knn = KNeighborsClassifier()
# knn.fit(X_train,y_train)

# y_predicted = knn.predict(X_test)

# print(accuracy_score(y_predicted,y_test))