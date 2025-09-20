import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_data

st.title('Sales Forecasting App')

uploaded_file = st.file_uploader('Upload your sales data CSV', type='csv')
if uploaded_file:
    df = load_data(uploaded_file)
    st.write('Data Preview:', df.head())
    st.write('Data Shape:', df.shape)
    st.write('Columns:', df.columns.tolist())
    st.write('Missing Values:', df.isnull().sum())

    st.subheader('Sales Volume Over Time')
    fig, ax = plt.subplots()
    df['Date'] = pd.to_datetime(df['Date'])
    df.groupby('Date')['Sales_Volume'].sum().plot(ax=ax)
    ax.set_ylabel('Sales Volume')
    st.pyplot(fig)

    st.subheader('Sales by Product Category')
    fig2, ax2 = plt.subplots()
    df.groupby('Product_Category')['Sales_Volume'].sum().plot(kind='bar', ax=ax2)
    ax2.set_ylabel('Total Sales Volume')
    st.pyplot(fig2)

    st.subheader('Sales by Store Location')
    fig3, ax3 = plt.subplots()
    df.groupby('Store_Location')['Sales_Volume'].sum().plot(kind='bar', ax=ax3)
    ax3.set_ylabel('Total Sales Volume')
    st.pyplot(fig3)

    # --- Forecasting Section ---
    st.subheader('Sales Forecasting (XGBoost)')
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    # Feature engineering: use numeric/categorical features
    df_feat = df.copy()
    df_feat['Date'] = pd.to_datetime(df_feat['Date'])
    df_feat['Year'] = df_feat['Date'].dt.year
    df_feat['Month'] = df_feat['Date'].dt.month
    df_feat['Day'] = df_feat['Date'].dt.day
    df_feat = pd.get_dummies(df_feat, columns=['Product_Category', 'Store_Location'], drop_first=True)

    X = df_feat.drop(['Sales_Volume', 'Date'], axis=1)
    y = df_feat['Sales_Volume']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f'Mean Absolute Error on Test Set: {mae:.2f}')

    # Simple future forecast: predict for next N days
    st.subheader('Forecast Future Sales')
    forecast_days = st.slider('Select forecast horizon (days)', 1, 30, 7)
    last_row = df_feat.iloc[-1:]
    future_dates = pd.date_range(df_feat['Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
    future_df = pd.DataFrame({
        'Year': future_dates.year,
        'Month': future_dates.month,
        'Day': future_dates.day
    })
    # Use last known values for categorical features
    for col in X.columns:
        if col not in future_df.columns:
            future_df[col] = last_row[col].values[0]
    future_pred = model.predict(future_df[X.columns])
    st.line_chart(pd.DataFrame({'Date': future_dates, 'Forecasted_Sales': future_pred}).set_index('Date'))
