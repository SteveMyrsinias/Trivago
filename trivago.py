import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os

missing_values = ["n/a", "na", "--", "?"] # pandas only detect NaN, NA,  n/a and values and empty shell
my_path = os.path.abspath(os.path.dirname(__file__))
df_train=pd.read_csv(r''+my_path+'\\data\\train.csv', sep=',', nrows=1000,na_values=missing_values)
# df_test=pd.read_csv(r''+my_path+'\\data\\test.csv', sep=',', nrows=1000,na_values=missing_values)

indexs = df_train[ (df_train['action_type']=='search for poi') | (df_train['action_type']=='filter selection') | (df_train['action_type']=='search for destination') | (df_train['action_type']=='change of sort order') ].index
df_train.drop(indexs, inplace=True)

df_train.drop(['city','device','current_filters','impressions','prices','platform'], axis=1, inplace=True)  

df_train = df_train[pd.notnull(df_train['reference'])]
df_train = df_train[pd.notnull(df_train['step'])]
df_train = df_train[pd.notnull(df_train['action_type'])]

grouped  = df_train.groupby(['user_id'])

for name,group in grouped:
   # print(group['timestamp'])
   # print(name)
   print(group)

# indexs = df_train[ (df_train['action_type']=='search for poi') | (df_train['action_type']=='filter selection') | (df_train['action_type']=='search for destination') | (df_train['action_type']=='change of sort order') ].index
# df_train.drop(indexs, inplace=True)
# print(df_train.shape)

# df_train = df_train[pd.notnull(df_train['reference'])]
# df_train = df_train[pd.notnull(df_train['step'])]
# df_train = df_train[pd.notnull(df_train['action_type'])]
# print(df_train.shape)
# print('ref', df_train['reference'].isnull().sum())
# print('step', df_train['step'].isnull().sum())
# print('action_type', df_train['action_type'].isnull().sum())
# users = df_train['user_id'].nunique()
# print('Users: ', users)

# reference = df_train['reference'].nunique()
# print('reference: ',reference)

# sessions = df_train['session_id'].nunique()
# print('Sessions: ',sessions)

# actionType = df_train['action_type'].nunique()
# print('Action Type: ',actionType)

# myvar  = df_train.groupby(['action_type'])['action_type'].count()
# print(myvar)
# print(type(myvar))
# # y = np.array([myvar[0], myvar[1], myvar[2], myvar[3], myvar[4], myvar[5], myvar[6], myvar[7] ])
# pieCharLabeles = ['clickout item ','interaction item deals','interaction item image','interaction item info','interaction item rating','search for item']
# plt.pie(myvar.values, labels=pieCharLabeles)
# plt.show()

# print(df_train.groupby(df_train['user_id'],as_index=False).size())
# s = df_train.groupby(df_train['user_id'],as_index=False).size()
# print(type(s))
# print('max ', s.max())
# print(type(s.max()))
# print('min ', s.min())
# print('mean ', s.mean())
# num_bins = 5
# # plt.hist(s.max().values, num_bins, facecolor='blue', alpha=0.5)
# # plt.show()

# df_train = pd.get_dummies(df_train, columns=['action_type'], prefix=['action_type_is'] )

# print(df_train.columns)

# df_train.drop(['user_id','session_id','timestamp','platform','city', 'device', 'current_filters','prices', 'impressions'], axis=1, inplace=True)  
# print(df_train.columns)
# X_df = df_train[['step', 'action_type_is_clickout item','action_type_is_interaction item deals','action_type_is_interaction item image','action_type_is_interaction item info','action_type_is_interaction item rating','action_type_is_search for item']]
# y_df = df_train[ 'reference']

# X_train, X_test, y_train, y_test = train_test_split(X_df,y_df,test_size=.2,random_state=42)


# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

# knn = KNeighborsClassifier()
# knn.fit(X_train,y_train)

# y_predicted = knn.predict(X_test)

# print(y_predicted)
# # Evaluate Model
# # print("confusion_matrix\n",confusion_matrix(y_test,y_predicted))
# # print("f1_score\n",f1_score(y_test,y_predicted))
# print("accuracy_score\n",accuracy_score(y_test,y_predicted))