import os
import statistics 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime as dt

# colums for the dataset that we will generate
columns = ['durationOfSession', 'steps', 'device', 'city', 'country', 'platform', 'current_filters', 'impressions', 'priceMean', 'hotel Facilities','interaction item image','search for poi','filter selection', 'interaction item info', 'search for destination', 'interaction item rating', 'change of sort order', 'interaction item deals', 'search for item', 'target']
generated_df = pd.DataFrame(columns=columns)

missing_values = ["n/a", "na", "--", "?"] # pandas only detect NaN, NA,  n/a and values and empty shell
my_path = os.path.abspath(os.path.dirname(__file__)) # get the current path
df_train=pd.read_csv(r''+my_path+'\\data\\train.csv', sep=',', na_values=missing_values) #, nrows=100000
itemMetaData=pd.read_csv(r''+my_path+'\\data\\item_metadata.csv', sep=',', na_values=missing_values)
print('initial shape: ',df_train.shape) # (1048575, 12)
groupedByUserId  = df_train.groupby(['user_id']) # group rows per user_id

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
   hotelFacilities=impressions=prices=currentFilters=interactionItemImage=searchForPoi=filterSelection=interactionItemInfo=searchForDestination=interactionItemRating=changeOfSortOrder=interactionItemDeals=searchForItem=0

   actionType = group['action_type'].value_counts() # get the action_type per session and count every action of a person
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
   
   currentFiltersGroup = group['current_filters'].value_counts() # get the sum of all the current_filter of a session
   for i, v in currentFiltersGroup.items():
      filters=i.split('|')
      currentFilters=len(filters)

   pricesGroup = group['prices'].value_counts() # get the median of all the prices of a session
   for i, v in pricesGroup.items():
      filterPrices=i.split('|')
      prices=statistics.median(map(int, filterPrices))

   impressionsGroup = group['impressions'].value_counts() # get the sum of all the impressions of a session
   for i, v in impressionsGroup.items():
      filters=i.split('|')
      for t in filters:
         t = int(t)
         hotelFacilityGroup = itemMetaData.loc[itemMetaData['item_id'] == t].value_counts() # get the sum of all hotel's facility fromt the metadata csv
         for j, k in hotelFacilityGroup.items():
            facilities= j[1].split('|')
            hotelFacilities += len(facilities)
      impressions=len(filters)

   # add values to the generated data frame
   generated_df.loc[generated_df.shape[0]] = [durationOfSession,steps,device,city,country,platform,currentFilters,impressions,prices,hotelFacilities,interactionItemImage,searchForPoi, filterSelection, interactionItemInfo, searchForDestination, interactionItemRating, changeOfSortOrder, interactionItemDeals, searchForItem,target]

# create a csv file 
generated_df.to_csv (r''+my_path+'\\data\\export_dataframe.csv', index = False, header=True)