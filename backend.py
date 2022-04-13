import pandas as pd
def get_supply_chain():
    supplyChain = pd.read_csv('supplyChain.csv')
    return supplyChain
  
def get_events():
    events = pd.read_csv('events.csv')
    return events
  
import os
os.chdir("C:/Users/bnhas/OneDrive/Desktop/Classes spring 2022/OMS Analytics/Data and Visual Analytics/Project/")
def importData(EventFile='events.csv',SupplyFile='sc.csv',SeverityValue=1.65,radius=2):
   import pandas as pd
   import numpy as np
   from datetime import date
   import math
   todays_date = date.today()
   Currentyear=int(todays_date.year)-1
   EventDataframe=pd.read_csv(EventFile)
   SCDataFrame=pd.read_csv(SupplyFile)
   EventwithDummies=pd.get_dummies(EventDataframe, columns=['violence_type','dyad_name','side_a','side_b','country','region'])
   SupplyWithDummies=pd.get_dummies(SCDataFrame, columns=['orig_ISO','dest_ISO','route_type'])
   SupplyWithDummies['routeBeginLat']=SupplyWithDummies['route_points'].apply(lambda x:np.asarray(x)[0][0][0])
   SupplyWithDummies['routeBeginLong']=SupplyWithDummies['route_points'].apply(lambda x:np.asarray(x)[0][0][1])
   EventwithDummies['year'] = pd.DatetimeIndex(EventwithDummies['date_start']).year
   col_list= list(SupplyWithDummies)
   col_list.remove('route_ID','orig_ISO','dest_ISO','route_points','routeBeginLong','routeBeginLat','weight','route_type')
   SupplyWithDummies['summedTrade']=SupplyWithDummies[col_list].sum(axis=1)
   IndexGrouped=SupplyWithDummies.groupby(level="route_ID").mean(axis=1).index
   MeanSummedTrade=SupplyWithDummies.groupby(level="route_ID").mean(axis=1)['summedTrade']
   StandardDeviationSummedTrade=SupplyWithDummies.groupby(level="route_ID").std(axis=1)['summedTrade']
   LowTrade=MeanSummedTrade.subtract(StandardDeviationSummedTrade*SeverityValue)
   LowTrade['route_id']=IndexGrouped
   SupplyWithDummies['Below95']=SupplyWithDummies.set_index('route_id').subtract(LowTrade.set_index('IndexGrouped'))
   SupplyWithDummies['LowTrade'] = 1 * (SupplyWithDummies['Below95']< 0)
   CurrentYearSupply=SupplyWithDummies[SupplyWithDummies['period']==Currentyear]
   CurrentYearRoutes=CurrentYearSupply['route_ID','route_points']
   CurrentYearRoutes['maxGood']=CurrentYearSupply[col_list].idxmax(axis=1)
   sumofSeverity=[]
   SumofConflictDuration=[]
   for index, row in SupplyWithDummies.iterrows():
    year=row['period']
    latitude=row['routeBeginLat']
    longitude=row['routeBeginLong']
    rows=EventDataframe[(math.sqrt((EventDataframe['latitude']-row['routeBeginLat'])**2+(EventDataframe['longitude']-row['routeBeginLong']))<= radius) &EventDataframe['year']==year]
    TotalSeverity=EventDataframe.sum(axis=1)['Intensity']
    TotalSeverity=EventDataframe.sum(axis=1)['conflict_duration_days']
    sumofSeverity.append(TotalSeverity)
   SupplyWithDummies['sumofSeverity']=sumofSeverity
   SupplyWithDummies['SumofConflictDuration']=SumofConflictDuration
   combinedyData=SupplyWithDummies['LowTrade']
   combinedXData=SupplyWithDummies['sumofSeverity','SumofConflictDuration','year','routeBeginLat','routeBeginLong']
   EarlierYearX=combinedXData[combinedXData['period']<Currentyear].to_numpy()
   CurrentYearX=combinedXData[combinedXData['period']==Currentyear].to_numpy()
   EarlierYearY=combinedyData[combinedyData['period']<Currentyear].to_numpy()
   CurrentYearY=combinedyData[combinedyData['period']==Currentyear].to_numpy()
   return EarlierYearX,CurrentYearX,EarlierYearY,CurrentYearY,CurrentYearRoutes

# List of Route #, Lat and Long List, Each good predicted to be harmed 

def MatchToRoutes(goodLookupFile,predictedData=v1,routes=CurrentYearRoutes):
    import numpy as np
    import pandas 
    GoodsDataFrame=pd.read_csv(goodLookupFile)
    mask=predictedData==1
    PredictedHarmedRoutes=CurrentYearRoutes[mask]
    GoodsandHarm=pd.concat([PredictedHarmedRoutes, GoodsDataFrame],on='product_code', axis=1, join="left")
    return GoodsandHarm
