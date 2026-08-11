import numpy as np
import pandas as pd
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

#Get data from training model
try:
    with open('House_price_model1.pkl','rb') as file:
        model_data=pickle.load(file)
    theta=model_data['theta_values']
    features_names=model_data['columns']
    scaler=model_data['scaler']
except Exception as e:
    st.error(f'There is error of training model:{e}')
    st.stop()

#Design web interface
st.image('dream_house.png')
st.set_page_config(page_title='House Price Predictor')
st.title('🏚️ House price prediction system')
st.write('Enter the details of your dream home below and get an approximate price estimate.')

st.markdown('---')

col1,col2=st.columns(2,gap='large')
df=pd.read_csv('House_price.csv')
all_districts=df['district'].dropna().unique().tolist()
all_areas=df['area'].dropna().unique().tolist()

with col1:
    st.subheader('Main details')
    district=st.sidebar.selectbox('◼ District',all_districts)
    year_built=st.sidebar.selectbox('◼ Year built',options=list(range(1985, 2031)))
    perch=st.number_input('◼ Number of perch',min_value=1.0,value=10.0,step=0.5)
    floors=st.number_input('◼ Number of floors',min_value=1,value=1)
    bedrooms=st.number_input('◼ Number of bedrooms',min_value=1,value=3)
    bathrooms=st.number_input('◼ Number of bathrooms',min_value=1,value=1)
    kitchen=st.number_input('◼ Area of kitchen(square feet)',min_value=50.0,value=150.0,step=10.0)

with col2:
    st.subheader('Extra details')
    parking_spots=st.number_input('◼ Number of parking spots',min_value=0,value=0)
    has_garden=st.radio('◼ Has garden',['Yes','No'])
    has_ac=st.radio('◼ Has AC',['Yes','No'])
    water=st.selectbox('◼ Water supply',['Pipe-borne','Well','Both'])
    electricity=st.selectbox('◼ Electricity',['Single phase','Three phase'])

#Prediction
if st.button('Calculate the price',type='primary',use_container_width=True):
    input_array=np.zeros(len(features_names))

    input_data={
        'perch':perch,
        'floors':floors,
        'bedrooms':bedrooms,
        'bathrooms':bathrooms,
        'kitchen_area_sqft':kitchen,
        'parking_spots':parking_spots,
        'year_built':year_built,
        'has_garden':1 if has_garden=='Yes' else 0,
        'has_ac':1 if has_ac=='Yes' else 0,
        f'district_{district}':1,
        f'water_supply_{water}':1,
        f'electricity_{electricity}':1
    }

    #include data to Array
    for index,col_name in enumerate(features_names):
        if col_name in input_data:
            input_array[index]=input_data[col_name]

    #Transforming Data
    scaled_input=scaler.transform(input_array.reshape(1,-1))

    #Add bias
    X_new=np.insert(scaled_input,0,1)


    prediction=X_new @ theta

    st.success(f'The predicted price of this house :**Rs.{prediction:,.2f}**')

#Graphical Representation
current_year=datetime.now().year
st.subheader(f"📊 Predicted Average House Prices by District for {current_year}")

try:
    df=pd.read_csv('House_price.csv')
    year_idx=features_names.index('year_built')

    yearly_price_increase=theta[year_idx + 1]/X_std[year_idx]

    district_data=df.groupby('district').agg({'price_lkr': 'mean', 'year_built': 'mean'})

    district_data['predicted_current_price']=district_data['price_lkr']+(
                current_year-district_data['year_built'])*yearly_price_increase

    predicted_prices=district_data['predicted_current_price'].sort_values(ascending=False)

    fig,ax=plt.subplots(figsize=(10,6))
    predicted_prices.plot(kind='bar',color='teal',ax=ax)

    ax.set_ylabel(f"Predicted Price in {current_year} (LKR)",fontsize=12)
    ax.set_xlabel("District",fontsize=12)
    plt.xticks(rotation=45,ha='right')

    st.pyplot(fig)
    st.markdown("---")

except Exception as e:
    st.warning(f"Could not generate graph. Please check the dataset. Error: {e}")
