# 🏚️ Sri Lanka House Price Predictor

A full-stack Machine Learning web application designed to predict real estate prices across various districts in Sri Lanka. This project demonstrates a dual-implementation approach to compare industry-standard libraries with fundamental mathematical modeling.

## 🚀 Key Features
* **Dual Implementations:** 
  * **From Scratch:** Linear Regression implemented using pure matrix mathematics (Normal Equation & Moore-Penrose pseudo-inverse).
  * **Library Approach:** Built using industry-standard `Scikit-learn` pipelines.
* **Interactive Web Interface:** Developed using **Streamlit** for real-time user inputs and instant price estimations.
* **Dynamic Data Visualization:** Automatically projects historical dataset trends to display current-year (2026) district-wise average house price comparisons using **Matplotlib**.
* **Robust Data Preprocessing:** Features feature scaling, one-hot encoding for categorical variables, and multicollinearity handling.

## 📊 Dataset Information
* **Source:** Real estate market records and housing listings in Sri Lanka (`House_price.csv`).
* **Features Included:** 
  * **Numerical:** Land extent in perches, number of floors, bedrooms, bathrooms, kitchen area ($\text{sqft}$), parking spots, and year built.
  * **Categorical:** District, water supply type, and electricity type.
  * **Target Variable:** House price in Sri Lankan Rupees ($\text{LKR}$).

## 🛠️ Built With
* **Python**
* **NumPy & Pandas** (Data manipulation & Matrix operations)
* **Scikit-learn** (Model evaluation & preprocessing)
* **Matplotlib** (Data visualization)
* **Streamlit** (Web framework)
*
