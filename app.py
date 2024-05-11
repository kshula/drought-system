import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error
from about import text
# Load your cleaned dataset with 'Month', 'SOI_index', and 'Weather' columns
@st.cache_data  # Cache the data for faster loading
def load_data():
    df = pd.read_csv('soi_data.csv')
    df['Month'] = pd.to_datetime(df['Month'])
    df.set_index('Month', inplace=True)
    return df

# Function to add lagged features to the dataset
def add_lagged_features(df, lag_periods=[1, 3]):
    for lag in lag_periods:
        df[f'SOI_index_lag_{lag}'] = df['SOI_index'].shift(lag)
    return df.dropna()

# Split data into train and test sets chronologically
def train_test_split(df, split_ratio=0.8):
    split_index = int(split_ratio * len(df))
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]
    return train_data, test_data

# Train and evaluate ARIMA model
def train_evaluate_arima(train_data, test_data, order=(5, 1, 0)):
    model = ARIMA(train_data['SOI_index'], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test_data))
    mse = mean_squared_error(test_data['SOI_index'], forecast)
    return mse

# Function to forecast weather states using ARIMA
def forecast_arima(df, forecast_period):
    model = ARIMA(df['SOI_index'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_period)
    return forecast

# Function to forecast weather states using SARIMA
def forecast_sarima(df, forecast_period):
    model = SARIMAX(df['SOI_index'], order=(1, 1, 1), seasonal_order=(0, 1, 1, 12))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_period)
    return forecast

# Function to forecast weather states using Random Forest and Gradient Boosting
def forecast_ml(df, forecast_period):
    # Prepare data for classification
    df['Weather_Category'] = np.where(df['SOI_index'] <= -0.469, 'El Nino',
                                       np.where(df['SOI_index'] >= 0.469, 'La Nina', 'La Nada (Neutral)'))

    X = df[['SOI_index']].values
    y = df['Weather_Category'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Random Forest Classifier
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_forecast = rf_model.predict(X_test[:forecast_period])

    # Gradient Boosting Classifier
    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train, y_train)
    gb_forecast = gb_model.predict(X_test[:forecast_period])

    return rf_forecast, gb_forecast

# Function to generate Random Forest forecast
def forecast_random_forest(df, forecast_period):
    X = df.index.factorize()[0].reshape(-1, 1)
    y = df['SOI_index']
    model = RandomForestRegressor()
    model.fit(X, y)
    forecast_index = pd.date_range(start=df.index[-1], periods=forecast_period + 1, freq='M')[1:]
    forecast = model.predict(forecast_index.factorize()[0].reshape(-1, 1))
    return forecast


# Function to calculate lengths of consecutive weather periods and classify weather categories
def calculate_periods(df):
    periods = df['Weather'].ne(df['Weather'].shift()).cumsum()
    weather_periods = df.groupby([periods])['SOI_index'].agg(['count', 'first', 'last']).reset_index()
    weather_periods['Weather_Category'] = weather_periods.apply(classify_weather_category, axis=1)
    return weather_periods


# Function to classify weather category based on first and last values
def classify_weather_category(row):
    first_value = row['first']
    last_value = row['last']
    
    if first_value <= -0.469 and last_value <= -0.469:
        return 'El Nino'
    elif first_value >= 0.469 and last_value >= 0.469:
        return 'La Nina'
    else:
        return 'La Nada (Neutral)'

# Modelling page
def modelling():
    st.title("Time Series Modelling")

    # Load data
    df = load_data()

    # Add lagged features to the dataset
    df = add_lagged_features(df, lag_periods=[1, 3])

    # Split data into train and test sets
    train_data, test_data = train_test_split(df)

    # Train and evaluate ARIMA model
    st.subheader("ARIMA Model Evaluation")
    arima_mse = train_evaluate_arima(train_data, test_data, order=(5, 1, 0))
    st.write(f"ARIMA MSE: {arima_mse}")

    # Plot SOI_index vs. time with colored annotations for weather categories
    st.subheader("SOI_index vs. Time with Weather Categories")
    fig_soi_weather = px.line(df, x=df.index, y='SOI_index', title='SOI_index vs. Time', color='Weather')
    st.plotly_chart(fig_soi_weather)

# Home page
def home():
    st.title("Niño-Nada-Niña: Drought Alert System")
    st.write("Explore seasonal patterns of the Southern Oscillation Index (SOI) and weather categories.")
    # Load data
    df = load_data()

    # First row of graphs
    st.subheader("SOI_index vs. Time with Weather Categories")
    fig_soi_weather = px.line(df, x=df.index, y='SOI_index', title='SOI_index vs. Time', color='Weather')
    st.plotly_chart(fig_soi_weather, use_container_width=True)

    # Second row of graphs
    weather_categories = df['Weather'].unique()
    for category in weather_categories:
        df_category = df[df['Weather'] == category]
        result_category = seasonal_decompose(df_category['SOI_index'], model='additive', period=12)
        st.subheader(f"Seasonal Decomposition of SOI_index during {category}")
        fig_category = px.line(result_category.trend, x=result_category.trend.index, y=result_category.trend.values, title=f'Trend - {category}')
        st.plotly_chart(fig_category, use_container_width=True)

    # Third row: Table and descriptive statistics
    weather_periods_data = calculate_periods(df)
    st.subheader("Weather Event Periods with Weather Categories")
    st.write(weather_periods_data[['count', 'first', 'last', 'Weather_Category']])

    st.subheader("Descriptive Statistics of Count per Period")
    st.write(weather_periods_data['count'].describe())

# Seasonal analysis page
def seasonal_analysis():
    st.title("Seasonal Analysis")

    # Load data
    df = load_data()

    # Perform seasonal decomposition for SOI_index
    result_soi = seasonal_decompose(df['SOI_index'], model='additive', period=12)  # Assuming annual seasonality (period=12 months)

    # Plot seasonal decomposition of SOI_index
    st.subheader("Seasonal Decomposition of SOI_index")
    fig_soi = px.line(result_soi.trend, x=result_soi.trend.index, y=result_soi.trend.values, title='Trend')
    st.plotly_chart(fig_soi)

    # Perform seasonal decomposition for each weather category
    weather_categories = df['Weather'].unique()
    for category in weather_categories:
        df_category = df[df['Weather'] == category]
        result_category = seasonal_decompose(df_category['SOI_index'], model='additive', period=12)
        st.subheader(f"Seasonal Decomposition of SOI_index during {category}")
        fig_category = px.line(result_category.trend, x=result_category.trend.index, y=result_category.trend.values, title=f'Trend - {category}')
        st.plotly_chart(fig_category)

# Periods page
def weather_periods():
    st.title("Weather Event Periods")

    # Load data
    df = load_data()

    # Calculate weather event periods and classify weather categories
    weather_periods_data = calculate_periods(df)

    # Display the calculated periods and weather categories in a table
    st.write("Weather Event Periods with Weather Categories:")
    st.write(weather_periods_data[['count', 'first', 'last', 'Weather_Category']])

    # Display descriptive statistics of count per period
    st.write("Descriptive Statistics of Count per Period:")
    st.write(weather_periods_data['count'].describe())

# Predictions page
def predictions():
    st.title("Weather Predictions")

    # Load data
    df = load_data()

    # Sidebar to choose forecast period
    forecast_period = st.sidebar.slider("Select Forecast Period (months)", min_value=1, max_value=12, value=6)

    # Generate Random Forest forecast
    rf_forecast = forecast_random_forest(df, forecast_period)

    # Ensure forecast has the same index as the forecast period
    last_date = df.index[-1]
    forecast_index = pd.date_range(start=last_date, periods=forecast_period + 1, freq='M')[1:]

    # Create DataFrame for the forecast with weather categories
    weather_categories = []
    for value in rf_forecast:
        if value <= -0.469:
            weather_categories.append('El Nino')
        elif value >= 0.469:
            weather_categories.append('La Nina')
        else:
            weather_categories.append('La Nada')

    rf_forecast_df = pd.DataFrame({
        'Forecast': rf_forecast,
        'Weather Category': weather_categories
    }, index=forecast_index)

    # Display forecast DataFrame with weather categories
    st.subheader("Random Forest Forecast with Weather Categories")
    st.write(rf_forecast_df)

    # Plot forecast with Plotly line chart (categorized by weather types)
    fig_forecast_categorized = px.line(rf_forecast_df, x=rf_forecast_df.index, y='Forecast', color='Weather Category',
                                       title=f"Forecast for Next {forecast_period} Months using Random Forest (Categorized)")
    st.plotly_chart(fig_forecast_categorized)

    # Plot forecast with Plotly line chart (without categorization)
    fig_forecast_uncategorized = px.line(rf_forecast_df, x=rf_forecast_df.index, y='Forecast',
                                          title=f"Forecast for Next {forecast_period} Months using Random Forest (Uncategorized)")
    st.plotly_chart(fig_forecast_uncategorized)

# Sidebar navigation with radio buttons for page selection
def sidebar_navigation():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", ["Home","About", "Seasonal Analysis", "Periods", "Modelling", "Predictions"])
    return page

def about():
    st.title("About")
    st.write(text)


# Main function to run the app
def main():
    # Sidebar navigation
    page = sidebar_navigation()

    if page == "Home":
        home()
    elif page == "Seasonal Analysis":
        seasonal_analysis()
    elif page == "Periods":
        weather_periods()
    elif page == "Modelling":
        modelling()
    elif page == "Predictions":
        predictions()
    elif page == "About":
        about()


# Run the app
if __name__ == "__main__":
    main()
