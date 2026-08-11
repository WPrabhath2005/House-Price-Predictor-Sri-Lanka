import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read CSV file
df=pd.read_csv('House_price.csv')

#Remove empty cells
df=df.dropna()

#Select subset of columns
df=df[['district','perch','floors','bedrooms','bathrooms','kitchen_area_sqft','parking_spots','has_garden','has_ac','water_supply','electricity','price_lkr','year_built']]

#Convert boolean to 1,0
df['has_garden']=df['has_garden'].astype(int)
df['has_ac']=df['has_ac'].astype(int)

#Convert letters/symbols(One-Hot Encoding)
#make new columns
df=pd.get_dummies(df,columns=['district','water_supply','electricity'],drop_first=True)

#Get X and Y matrices
Y_data=df['price_lkr'].values.astype(float)
X_data=df.drop(['price_lkr'],axis=1).values.astype(float)

#Feature Scaling(Standardization)
X_mean=np.mean(X_data,axis=0)
X_std=np.std(X_data,axis=0)
X_data=(X_data-X_mean)/X_std

#Add bias column
m=len(X_data)
X_b=np.c_[np.ones((m,1)),X_data]

#Find weights matrix
theta=np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ Y_data

#get parameters of all columns
features_names=df.drop(['price_lkr'],axis=1).columns

features_list=list(features_names)

model_data={'theta_values':theta,'columns':features_list,
            'mean':X_mean,'std':X_std}

with open('House_price_model.pkl','wb') as file:
    pickle.dump(model_data,file)