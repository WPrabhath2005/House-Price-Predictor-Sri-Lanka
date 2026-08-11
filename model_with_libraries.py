import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pickle

#Read data
df=pd.read_csv('House_price.csv')
df=df.drop('area',axis=1)

#Convert to 1 and 0 letters/symbols(One-Hot Encoding)
df_encoded=pd.get_dummies(df,columns=['district','water_supply','electricity'],drop_first=True)
X=df_encoded.drop('price_lkr',axis=1)
y=df_encoded['price_lkr']

#Scaling data
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#Train Model
model=LinearRegression()
model.fit(X_scaled,y)

#Get weights and parameters and save model
model_data={
    'theta_values':np.insert(model.coef_,0,model.intercept_),
    'columns':X.columns.tolist(),
    'scaler':scaler
}
with open('House_price_model1.pkl','wb') as file:
    pickle.dump(model_data,file)
