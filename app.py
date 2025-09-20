import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_data
import numpy as np

st.title('Sales Forecasting App')

uploaded_file = st.file_uploader('Upload your sales data CSV', type='csv')
if uploaded_file:
    df = load_data(uploaded_file)
    st.write('Data Preview:', df.head())
    st.write('Data Shape:', df.shape)
    st.write('Columns:', df.columns.tolist())
    st.write('Missing Values:', df.isnull().sum())

    # --- Flexible column selection for different CSVs (optional columns) ---
    st.subheader('Column Mapping')
    columns = df.columns.tolist()
    date_col = st.selectbox('Select the Date column', columns, index=columns.index('Date') if 'Date' in columns else 0)
    sales_col = st.selectbox('Select the Sales Volume column', columns, index=columns.index('Sales_Volume') if 'Sales_Volume' in columns else 0)
    # Optional columns
    category_col = st.selectbox('Select the Product Category column (optional)', ['None'] + columns, index=(columns.index('Product_Category')+1) if 'Product_Category' in columns else 0)
    location_col = st.selectbox('Select the Store Location column (optional)', ['None'] + columns, index=(columns.index('Store_Location')+1) if 'Store_Location' in columns else 0)

    # Rename columns for internal consistency
    rename_dict = {date_col: 'Date', sales_col: 'Sales_Volume'}
    if category_col != 'None':
        rename_dict[category_col] = 'Product_Category'
    if location_col != 'None':
        rename_dict[location_col] = 'Store_Location'
    df = df.rename(columns=rename_dict)

    # Show column mapping preview before continuing
    st.info(f"Column mapping: Date → '{date_col}', Sales Volume → '{sales_col}', Product Category → '{category_col if category_col != 'None' else 'N/A'}', Store Location → '{location_col if location_col != 'None' else 'N/A'}'")

    # Check for required columns
    required = ['Date', 'Sales_Volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}. Please check your file and column selections.\n\nColumns in your file: {', '.join(df.columns)}")
        st.stop()

    # Fill missing optional columns with default values if not present
    if 'Product_Category' not in df.columns:
        df['Product_Category'] = 'Unknown'
    if 'Store_Location' not in df.columns:
        df['Store_Location'] = 'Unknown'

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

    # --- Feature engineering: add advanced features ---
    df_feat = df.copy()
    df_feat['Date'] = pd.to_datetime(df_feat['Date'])
    df_feat = df_feat.sort_values('Date')
    # Lag features
    df_feat['Sales_Lag_1'] = df_feat['Sales_Volume'].shift(1)
    df_feat['Sales_Lag_3'] = df_feat['Sales_Volume'].shift(3)
    df_feat['Sales_Lag_7'] = df_feat['Sales_Volume'].shift(7)
    df_feat['Sales_Lag_14'] = df_feat['Sales_Volume'].shift(14)
    # Rolling statistics
    df_feat['Sales_Rolling_7_mean'] = df_feat['Sales_Volume'].rolling(window=7).mean()
    df_feat['Sales_Rolling_7_std'] = df_feat['Sales_Volume'].rolling(window=7).std()
    df_feat['Sales_Rolling_7_min'] = df_feat['Sales_Volume'].rolling(window=7).min()
    df_feat['Sales_Rolling_7_max'] = df_feat['Sales_Volume'].rolling(window=7).max()
    # Cumulative sum (year-to-date)
    df_feat['Year'] = df_feat['Date'].dt.year
    df_feat['YTD_Sales'] = df_feat.groupby('Year')['Sales_Volume'].cumsum()
    # Date features
    df_feat['Month'] = df_feat['Date'].dt.month
    df_feat['Day'] = df_feat['Date'].dt.day
    df_feat['Is_Weekend'] = df_feat['Date'].dt.weekday >= 5
    df_feat['Is_Month_Start'] = df_feat['Date'].dt.is_month_start
    df_feat['Is_Month_End'] = df_feat['Date'].dt.is_month_end
    # Interaction features
    df_feat['Promo_x_Price'] = df_feat['Promotion'] * df_feat['Price']
    df_feat['Promo_x_Store_Urban'] = df_feat['Promotion'] * (df_feat['Store_Location'] == 'Urban').astype(int)
    # Categorical dummies
    df_feat = pd.get_dummies(df_feat, columns=['Product_Category', 'Store_Location'], drop_first=True)
    df_feat = df_feat.dropna()  # Drop rows with NaN from lag/rolling

    X = df_feat.drop(['Sales_Volume', 'Date'], axis=1)
    y = df_feat['Sales_Volume']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.write(f'Mean Absolute Error on Test Set: {mae:.2f}')

    # --- Future forecast with advanced features ---
    st.subheader('Forecast Future Sales')
    forecast_days = st.slider('Select forecast horizon (days)', 1, 30, 7)
    last_rows = df_feat.iloc[-14:].copy()  # For lags/rolling
    future_preds = []
    future_dates = []
    for i in range(forecast_days):
        next_date = last_rows['Date'].max() + pd.Timedelta(days=1)
        row = last_rows.iloc[-1:].copy()
        row['Date'] = next_date
        row['Year'] = next_date.year
        row['Month'] = next_date.month
        row['Day'] = next_date.day
        row['Is_Weekend'] = next_date.weekday() >= 5
        row['Is_Month_Start'] = next_date.is_month_start
        row['Is_Month_End'] = next_date.is_month_end
        # Lag features
        row['Sales_Lag_1'] = last_rows['Sales_Volume'].iloc[-1]
        row['Sales_Lag_3'] = last_rows['Sales_Volume'].iloc[-3] if len(last_rows) >= 3 else last_rows['Sales_Volume'].iloc[0]
        row['Sales_Lag_7'] = last_rows['Sales_Volume'].iloc[-7] if len(last_rows) >= 7 else last_rows['Sales_Volume'].iloc[0]
        row['Sales_Lag_14'] = last_rows['Sales_Volume'].iloc[0]
        # Rolling stats
        row['Sales_Rolling_7_mean'] = last_rows['Sales_Volume'].rolling(window=7).mean().iloc[-1]
        row['Sales_Rolling_7_std'] = last_rows['Sales_Volume'].rolling(window=7).std().iloc[-1]
        row['Sales_Rolling_7_min'] = last_rows['Sales_Volume'].rolling(window=7).min().iloc[-1]
        row['Sales_Rolling_7_max'] = last_rows['Sales_Volume'].rolling(window=7).max().iloc[-1]
        # Cumulative sum
        row['YTD_Sales'] = last_rows['Sales_Volume'].sum() + row['Sales_Lag_1']
        # Interaction features
        row['Promo_x_Price'] = row['Promotion'] * row['Price']
        row['Promo_x_Store_Urban'] = row['Promotion'] * row.get('Store_Location_Urban', 0)
        # Keep categorical dummies
        for col in X.columns:
            if col not in row.columns:
                row[col] = last_rows.iloc[-1][col]
        pred = model.predict(row[X.columns])[0]
        row['Sales_Volume'] = pred
        last_rows = pd.concat([last_rows, row], ignore_index=True)
        last_rows = last_rows.iloc[1:]  # Keep last 14
        future_preds.append(pred)
        future_dates.append(next_date)
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted_Sales': future_preds})
    st.line_chart(forecast_df.set_index('Date'))
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='Download Forecast as CSV',
        data=csv,
        file_name='sales_forecast.csv',
        mime='text/csv'
    )

    # --- Model Explainability: Feature Importance ---
    st.subheader('Feature Importance (XGBoost)')
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values('Importance', ascending=False)
    fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(15), ax=ax_imp)
    ax_imp.set_title('Top 15 Feature Importances')
    st.pyplot(fig_imp)

    # --- Model Validation: Cross-Validation and Error Metrics ---
    st.subheader('Model Validation (Cross-Validation)')
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score
    cv_mae = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    cv_rmse = cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
    st.write(f'Cross-validated MAE: {-cv_mae.mean():.2f}')
    st.write(f'Cross-validated RMSE: {-cv_rmse.mean():.2f}')

    # --- Actual vs. Predicted Plot ---
    st.subheader('Actual vs. Predicted Sales (Test Set)')
    fig_pred, ax_pred = plt.subplots()
    ax_pred.scatter(y_test, y_pred, alpha=0.5)
    ax_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax_pred.set_xlabel('Actual Sales')
    ax_pred.set_ylabel('Predicted Sales')
    ax_pred.set_title('Actual vs. Predicted Sales')
    st.pyplot(fig_pred)

    # --- Error Metrics Table ---
    st.subheader('Error Metrics (Test Set)')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.write(pd.DataFrame({
        'MAE': [mae],
        'RMSE': [rmse],
        'R2': [r2]
    }))

    # --- Advanced Visualizations ---
    st.header('Advanced Visualizations')

    # 1. Seasonality Heatmap (Sales by Day of Week and Month)
    st.subheader('Seasonality Heatmap (Day of Week vs. Month)')
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    pivot = df.pivot_table(index='DayOfWeek', columns='Month', values='Sales_Volume', aggfunc='mean')
    fig_season, ax_season = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu', ax=ax_season)
    ax_season.set_title('Average Sales by Day of Week and Month')
    ax_season.set_xlabel('Month')
    ax_season.set_ylabel('Day of Week (0=Mon)')
    st.pyplot(fig_season)
    st.download_button('Download Seasonality Heatmap', data=fig_season_to_bytes(fig_season), file_name='seasonality_heatmap.png', mime='image/png')

    # 2. Promotion Impact Plot (if Promotion column exists)
    if 'Promotion' in df.columns:
        st.subheader('Promotion Impact on Sales')
        fig_promo, ax_promo = plt.subplots()
        sns.boxplot(x='Promotion', y='Sales_Volume', data=df, ax=ax_promo)
        ax_promo.set_title('Sales Distribution by Promotion')
        st.pyplot(fig_promo)
        st.download_button('Download Promotion Impact Plot', data=fig_season_to_bytes(fig_promo), file_name='promotion_impact.png', mime='image/png')

    # 3. Rolling Average Plot
    st.subheader('Sales Volume with 7-Day Rolling Average')
    fig_roll, ax_roll = plt.subplots()
    df_sorted = df.sort_values('Date')
    ax_roll.plot(df_sorted['Date'], df_sorted['Sales_Volume'], label='Daily Sales')
    ax_roll.plot(df_sorted['Date'], df_sorted['Sales_Volume'].rolling(window=7).mean(), label='7-Day Rolling Mean', color='orange')
    ax_roll.set_ylabel('Sales Volume')
    ax_roll.set_title('Sales Volume and Rolling Average')
    ax_roll.legend()
    st.pyplot(fig_roll)
    st.download_button('Download Rolling Average Plot', data=fig_season_to_bytes(fig_roll), file_name='rolling_average.png', mime='image/png')

    # 4. Category/Location Trends Over Time
    st.subheader('Sales by Product Category Over Time')
    if 'Product_Category' in df.columns:
        fig_cat, ax_cat = plt.subplots()
        for cat in df['Product_Category'].unique():
            sub = df[df['Product_Category'] == cat]
            ax_cat.plot(sub['Date'], sub['Sales_Volume'], label=cat)
        ax_cat.set_title('Sales by Product Category Over Time')
        ax_cat.set_ylabel('Sales Volume')
        ax_cat.legend()
        st.pyplot(fig_cat)
        st.download_button('Download Category Trends', data=fig_season_to_bytes(fig_cat), file_name='category_trends.png', mime='image/png')

    st.subheader('Sales by Store Location Over Time')
    if 'Store_Location' in df.columns:
        fig_loc, ax_loc = plt.subplots()
        for loc in df['Store_Location'].unique():
            sub = df[df['Store_Location'] == loc]
            ax_loc.plot(sub['Date'], sub['Sales_Volume'], label=loc)
        ax_loc.set_title('Sales by Store Location Over Time')
        ax_loc.set_ylabel('Sales Volume')
        ax_loc.legend()
        st.pyplot(fig_loc)
        st.download_button('Download Location Trends', data=fig_season_to_bytes(fig_loc), file_name='location_trends.png', mime='image/png')

    # 5. Outlier Detection Plot (optional)
    st.subheader('Outlier Detection in Sales Volume')
    q1 = df['Sales_Volume'].quantile(0.25)
    q3 = df['Sales_Volume'].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df['Sales_Volume'] < lower) | (df['Sales_Volume'] > upper)]
    fig_out, ax_out = plt.subplots()
    ax_out.plot(df['Date'], df['Sales_Volume'], label='Sales')
    ax_out.scatter(outliers['Date'], outliers['Sales_Volume'], color='red', label='Outliers')
    ax_out.set_title('Sales Volume with Outliers Highlighted')
    ax_out.set_ylabel('Sales Volume')
    ax_out.legend()
    st.pyplot(fig_out)
    st.download_button('Download Outlier Plot', data=fig_season_to_bytes(fig_out), file_name='outlier_plot.png', mime='image/png')

# --- Utility function for figure download ---
def fig_season_to_bytes(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf.read()
