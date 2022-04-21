import pandas as pd
def get_supply_chain():
    supplyChain = pd.read_csv('sc_data_2019.csv')
    return supplyChain
  
def get_events():
    events = pd.read_csv('events.csv')
    return events
  
import os
# os.chdir("C:/Users/bnhas/OneDrive/Desktop/Classes spring 2022/OMS Analytics/Data and Visual Analytics/Project/")
def importData(EventFile='events.csv',SupplyFile='sc_data_sample.csv',SeverityValue=.3,radius=50):
   import pandas as pd
   import numpy as np
   from datetime import date
   import math
   todays_date = date.today()
   Currentyear=int(todays_date.year)-1
   EventwithDummies=pd.read_csv(EventFile)
   SupplyWithDummies=pd.read_csv(SupplyFile)
   #EventwithDummies=pd.get_dummies(EventDataframe, columns=['violence_type','dyad_name','side_a','side_b','country','region'])
   #SupplyWithDummies=pd.get_dummies(SCDataFrame, columns=['orig_ISO','dest_ISO','route_type'])
   SupplyWithDummies['routeBeginLat']=SupplyWithDummies['route_points'].apply(lambda x:x.split(",")[0][2:])
   SupplyWithDummies['routeBeginLong']=SupplyWithDummies['route_points'].apply(lambda x:x.split(",")[1][:-3])
   EventwithDummies['year'] = pd.DatetimeIndex(EventwithDummies['date_start']).year
   col_list= list(SupplyWithDummies)
   col_list.remove('route_ID')
   col_list.remove('orig_ISO')
   col_list.remove('dest_ISO')
   col_list.remove('route_points')
   col_list.remove('routeBeginLong')
   col_list.remove('routeBeginLat')
   col_list.remove('weight')
   col_list.remove('route_type')
   SupplyWithDummies['summedTrade']=SupplyWithDummies[col_list].sum(axis=1)
   IndexGrouped=SupplyWithDummies.groupby(by="route_ID").mean().index
   MeanSummedTrade=SupplyWithDummies.groupby(by="route_ID").mean()['summedTrade']
   StandardDeviationSummedTrade=SupplyWithDummies.groupby(by="route_ID").std()['summedTrade']
   LowTrade=pd.DataFrame()
   LowTrade['lowTrade']=MeanSummedTrade.subtract(StandardDeviationSummedTrade*SeverityValue)
   LowTrade['route_id']=IndexGrouped
   for index, row in SupplyWithDummies.iterrows():
    routeID=row['route_ID']
    Below95=row['summedTrade']-(LowTrade[LowTrade['route_id']==routeID]['lowTrade'])
    row['LowTrade'] = 1 * (Below95< 0)
   CurrentYearSupply=SupplyWithDummies[SupplyWithDummies['Year']==Currentyear]
   CurrentYearRoutes=CurrentYearSupply[['route_ID','route_points']]
   CurrentYearRoutes['product_code']=CurrentYearSupply[col_list].idxmax(axis=1)
   sumofSeverity=[]
   SumofConflictDuration=[]
   for index, row in SupplyWithDummies.iterrows():
    year=row['Year']
    latitude=row['routeBeginLat']
    longitude=row['routeBeginLong']
    severityRow=0
    DurationRow=0
    for index1, row1 in EventwithDummies.iterrows():
        rowvalue=(math.sqrt((float(row1['latitude'])-float(row['routeBeginLat']))**2+(float(row1['longitude'])-float(row['routeBeginLong']))**2))
        if rowvalue<=radius and row1['year']==year:
            DurationRow=DurationRow+row1['conflict_duration_days']
            severityRow=severityRow+row1['scale']
    sumofSeverity.append(severityRow)
    SumofConflictDuration.append(DurationRow)
   SupplyWithDummies['sumofSeverity']=sumofSeverity
   SupplyWithDummies['SumofConflictDuration']=SumofConflictDuration
   combinedyData=SupplyWithDummies['LowTrade']
   combinedXData=SupplyWithDummies[['sumofSeverity','SumofConflictDuration','Year','routeBeginLat','routeBeginLong']]
   EarlierYearX=combinedXData[combinedXData['Year']<Currentyear].to_numpy()
   CurrentYearX=combinedXData[combinedXData['Year']==Currentyear].to_numpy()
   EarlierYearY=combinedyData[combinedyData['Year']<Currentyear].to_numpy()
   CurrentYearY=combinedyData[combinedyData['Year']==Currentyear].to_numpy()
   return EarlierYearX,CurrentYearX,EarlierYearY,CurrentYearY,CurrentYearRoutes

EarlierYearX,CurrentYearX,EarlierYearY,CurrentYearY,CurrentYearRoutes=importData()
x_dataTrain = torch.tensor(EarlierYearX)
x_dataTest = torch.tensor(CurrentYearX)
y_dataTrain = torch.tensor(EarlierYearY)
y_dataTest = torch.tensor(CurrentYearY)

import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid , save_image
batch_size=64
train_loader = torch.utils.data.DataLoader((x_dataTrain,y_dataTrain),batch_size=batch_size)
class RBM(nn.Module):
   def __init__(self,
               n_vis=4,
               n_hin=4,
               k=5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin,n_vis)*1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k
    
   def sample_from_p(self,p):
       return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))
    
   def v_to_h(self,v):
        p_h = F.sigmoid(F.linear(v,self.W,self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h,sample_h
    
   def h_to_v(self,h):
        p_v = F.sigmoid(F.linear(h,self.W.t(),self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v,sample_v
        
   def forward(self,v):
        pre_h1,h1 = self.v_to_h(v)
        
        h_ = h1
        for _ in range(self.k):
            pre_v_,v_ = self.h_to_v(h_)
            pre_h_,h_ = self.v_to_h(v_)
        
        return v,v_
    
   def free_energy(self,v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v,self.W,self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

rbm = RBM(k=1)
train_op = optim.SGD(rbm.parameters(),0.1)

for epoch in range(10):
    loss_ = []
    for _, (data,target) in enumerate(train_loader):
        data = Variable(data.view(-1,4))
        sample_data = x_dataTest
        v,v1 = rbm(sample_data)
        loss = rbm.free_energy(v) - rbm.free_energy(v1)
        loss_.append(loss.data)
        train_op.zero_grad()
        loss.backward()
        train_op.step()

# List of Route #, Lat and Long List, Each good predicted to be harmed 

def MatchToRoutes(predictedData=v1,routes=CurrentYearRoutes,goodLookupFile='HS6_Lookup.csv'):
    import numpy as np
    import pandas 
    GoodsDataFrame=pd.read_csv(goodLookupFile)
    mask=predictedData==1
    PredictedHarmedRoutes=CurrentYearRoutes[mask]
    GoodsandHarm=pd.concat([PredictedHarmedRoutes, GoodsDataFrame],on='product_code', axis=1, join="left")
    return GoodsandHarm

GoodsandHarm=MatchToRoutes()
print(GoodsandHarm)
