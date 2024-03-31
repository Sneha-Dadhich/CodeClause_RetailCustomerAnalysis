import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error as mae
from datetime import datetime
import calendar
import holidays
import warnings
warnings.filterwarnings("ignore")

# Function to read and preprocess data
def read_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    parts = df["date"].str.split("-", n=3, expand=True) 
    df["year"] = parts[0].astype('int') 
    df["month"] = parts[1].astype('int') 
    df["day"] = parts[2].astype('int')
    
    df['weekend'] = df.apply(lambda x: weekend_or_weekday(x['year'], x['month'], x['day']), axis=1)
    df['holidays'] = df['date'].apply(is_holiday) 
    df['m1'] = np.sin(df['month'] * (2 * np.pi / 12)) 
    df['m2'] = np.cos(df['month'] * (2 * np.pi / 12))
    df['weekday'] = df.apply(lambda x: which_day(x['year'], x['month'],x['day']), axis=1)
    df.drop('date', axis=1, inplace=True)
    return df

# Function to check if a date is a weekend or weekday
def weekend_or_weekday(year, month, day):
    d = datetime(year, month, day)
    if d.weekday() > 4: 
        return 1
    else: 
        return 0

# Function to check if a date is a holiday
def is_holiday(x): 
    india_holidays = holidays.country_holidays('IN') 
    if india_holidays.get(x): 
        return 1
    else: 
        return 0

# Function to calculate which day of the week
def which_day(year, month, day):       
    d = datetime(year, month, day) 
    return d.weekday()

# Function to visualize sales by categorical features
def visualize_categorical(df, features):
    fig, axs = plt.subplots(figsize=(20, 10), nrows=2, ncols=3)
    for i, col in enumerate(features): 
        ax = df.groupby(col).mean()['sales'].plot.bar(ax=axs[i//3, i%3])
        ax.set_xlabel(None)  # Hide x-axis label
        ax.get_xaxis().set_visible(False)  # Hide x-axis ticks
    st.pyplot(fig)

# Function to visualize sales by day
def visualize_sales_by_day(df):
    fig=plt.figure(figsize=(10,5)) 
    df.groupby('day').mean()['sales'].plot()
    st.pyplot(fig)

# Function to visualize sales and Simple Moving Average
def visualize_sales_and_sma(df):
    plt.figure(figsize=(15, 10))
    window_size = 30
    data = df[df['year']==2013] 
    windows = data['sales'].rolling(window_size)
    sma = windows.mean()
    sma = sma[window_size - 1:]

    plt.plot(data['sales'])
    plt.plot(sma)
    plt.legend(['Sales', 'Simple Moving Average'])
    st.pyplot()

# Function to visualize distribution and boxplot of sales
def visualize_sales_distribution(df):
    fig=plt.figure(figsize=(12, 5)) 
    plt.subplot(1, 2, 1) 
    sns.distplot(df['sales']) 

    plt.subplot(1, 2, 2) 
    sns.boxplot(df['sales']) 
    st.pyplot(fig)

# Function to visualize correlation heatmap
def visualize_correlation_heatmap(df):
    fig=plt.figure(figsize=(10, 10)) 
    sns.heatmap(df.corr() > 0.8, annot=True, cbar=False) 
    st.pyplot(fig)

# Function to train models and display errors
def train_and_display_errors(X_train, X_val, Y_train, Y_val):
    models = [LinearRegression(), XGBRegressor(), Lasso(), Ridge()] 
    for model in models: 
        model.fit(X_train, Y_train) 
        st.write(f'{model}: ') 
        train_preds = model.predict(X_train) 
        st.write('Training Error: ', mae(Y_train, train_preds)) 
        val_preds = model.predict(X_val) 
        st.write('Validation Error: ', mae(Y_val, val_preds)) 

# Main function
def main():
    #[global]
    suppressWarning = True


    st.title('Store Demand Analysis')
    # Preprocessing
    uploaded_file="C:\\Sneha\\Programs1\\Python\\Internship\\CodeClause\\DemandAnalysis\\StoreDemand.csv"
    df = read_and_preprocess_data(uploaded_file)
    features = ['store', 'year', 'month', 'weekday', 'weekend', 'holidays']

    visualization_option = st.sidebar.selectbox("Select Visualization", 
                                                     ["Categorical Features", "Sales by Day", 
                                                      "Sales and Simple Moving Average", "Sales Distribution",
                                                      "Correlation Heatmap"])

    if visualization_option == "Categorical Features":
        visualize_categorical(df, features)
    elif visualization_option == "Sales by Day":
        visualize_sales_by_day(df)
    elif visualization_option == "Sales Distribution":
        visualize_sales_distribution(df)
    elif visualization_option == "Correlation Heatmap":
        visualize_correlation_heatmap(df)

    df = df[df['sales'] < 140]
    features = df.drop(['sales', 'year'], axis=1)
    target = df['sales'].values

    X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.05, random_state=22)
    train_and_display_errors(X_train, X_val, Y_train, Y_val)

if __name__ == '__main__':
    main()
