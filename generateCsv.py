import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from datetime import datetime as dt
import os

# ToDo: add current_filters, impressions, prices,

missing_values = ["n/a", "na", "--", "?"] # pandas only detect NaN, NA,  n/a and values and empty shell
my_path = os.path.abspath(os.path.dirname(__file__))
df_train=pd.read_csv(r''+my_path+'\\data\\train.csv', sep=',', na_values=missing_values)
print(df_train.shape) #(1048575, 12)
groupedByUserId  = df_train.groupby(['user_id']) # group rows per user_id
# colums for the data frame that we generate
columns = ['durationOfSession', 'steps', 'device', 'city', 'country', 'platform','interaction item image','search for poi','filter selection', 'interaction item info', 'search for destination', 'interaction item rating', 'change of sort order', 'interaction item deals', 'search for item', 'target']
generated_df = pd.DataFrame(columns=columns)

# ToDo: tackle the case of 2 click-out per
for index,group in groupedByUserId:
   df = group.iloc[[0, -1]] # get the first and the last column of the group
   startTimeStamp = dt.fromtimestamp(df.iloc[0, 2]) # get the timestamp when the user starts the session, and convert it into datetime 
   endTimeStamp = dt.fromtimestamp(df.iloc[1, 2]) # get the timestamp when the user ends the session, and convert it into datetime
   durationOfSession = (endTimeStamp-startTimeStamp).total_seconds() # compute the duration of the session in seconds
   steps = df.iloc[1, 3] # get the steps that the user is procceded from the last row of the group
   device = df.iloc[1, 8] # get the user's device
   cityAndCountry = df.iloc[1, 7].split(',') # get the user's city and country
   city = cityAndCountry[0]
   country = cityAndCountry[1]
   platform = df.iloc[1, 6] # get user's platform 
   target=0
   if 'clickout item' in group['action_type'].values: # identify if the user procced in click-out (yes:1 , not:0)
      target=1
   
   # initialize multiple variables
   interactionItemImage=searchForPoi=filterSelection=interactionItemInfo=searchForDestination=interactionItemRating=changeOfSortOrder=interactionItemDeals=searchForItem=0

   actionType = group['action_type'].value_counts()
   for i, v in actionType.items():
      if i == 'interaction item image':
         interactionItemImage = v
      if i == 'search for poi':
         searchForPoi = v
      if i == 'filter selection':
         filterSelection = v
      if i == 'interaction item info':
         interactionItemInfo = v
      if i == 'search for destination':
         searchForDestination = v
      if i == 'interaction item rating':
         interactionItemRating = v
      if i == 'change of sort order':
         changeOfSortOrder = v
      if i == 'interaction item deals':
         interactionItemDeals = v
      if i == 'search for item':
         searchForItem = v
      #print('index: ', i, 'value: ', v)
   
   # add values to the generated data frame
   generated_df.loc[generated_df.shape[0]] = [durationOfSession,steps,device,city,country,platform,interactionItemImage,searchForPoi, filterSelection, interactionItemInfo, searchForDestination, interactionItemRating, changeOfSortOrder, interactionItemDeals, searchForItem,target]

# create a csv file 
generated_df.to_csv (r''+my_path+'\\data\\export_dataframe.csv', index = False, header=True)
#print(generated_df)