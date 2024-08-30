import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load data
data = pd.read_excel(r'C:\DJ\Mtech - Data Science\1st Sem\Financial Market\Project\Data\Data.xlsx')
data.set_index('Date', inplace=True)


# Plotting the graphs 
plt.figure(figsize=(12, 6))

plt.plot(data['NET_FLL'], label='Net Purchase FII', color='green')

plt.plot(data['NET_DLL'], label='Net Purchase FII', color='orange')
plt.title('Time Series Plot for Net Purchase for FIIs vs DIIs')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.show()

# Plotting Time Series for 'Close'
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Closing Price', color='blue')
plt.title('Time Series Plot for Closing Price of NIFTY')
plt.xlabel('Date')
plt.ylabel('Monthly Closing Values of NIFTY')
plt.legend()
plt.show()


# Function to check stationarity
def check_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    return result

# Check stationarity of 'Close' column
result_close = check_stationarity(data['Close'])
result_NET_FLL = check_stationarity(data['NET_FLL'])
result_NET_DLL = check_stationarity(data['NET_DLL'])


if result_close[1] > 0.05:
    data['Close_diff'] = data['Close'].diff().dropna()

    plt.plot(data['Close_diff'])
    plt.title('NIFTY After Differencing ')
    plt.show()

    result_diff = check_stationarity(data['Close_diff'])


exog_vars = ['NET_FLL', 'NET_DLL', 'Close_diff']
for var in exog_vars:
    data[var] = np.where(np.isinf(data[var]), np.nan, data[var])

    if data[var].isnull().any():
        data[var].fillna(data[var].mean(), inplace=True)

# Check stationarity of differenced series after handling missing values
result_diff = check_stationarity(data['Close_diff'])

# Plot ACF and PACF
plot_acf(data['Close_diff'], lags=20, title='Autocorrelation Function (ACF)')
plt.show()

plot_pacf(data['Close_diff'], lags=20, title='Partial Autocorrelation Function (PACF)')
plt.show()


train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]


order = (0, 1, 0) 
model = ARIMA(train['Close'], exog=train[exog_vars], order=order)
results = model.fit()


predictions = results.get_forecast(steps=len(test), exog=test[exog_vars])
predicted_values = predictions.predicted_mean


mse = mean_squared_error(test['Close'], predicted_values)
rmse = sqrt(mse)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")


plt.plot(train['Close'], label='Train')
plt.plot(test['Close'], label='Test')
plt.plot(test.index, predicted_values, label='Predicted', color='red')
plt.title('ARIMA Model: Actual vs. Predicted (Test Set)')
plt.legend()
plt.show()


model_full = ARIMA(data['Close'], exog=data[exog_vars], order=order)
results_full = model_full.fit()


forecast_steps = 12
forecast = results_full.get_forecast(steps=forecast_steps, exog=data.iloc[-forecast_steps:][exog_vars])
forecast_values = forecast.predicted_mean


forecast_dates = pd.date_range(start=data.index[-1] + pd.DateOffset(1), periods=forecast_steps, freq='M')
forecast_summary = pd.DataFrame({'Date': forecast_dates, 'Forecasted_Close': forecast_values})
forecast_summary.set_index('Date', inplace=True)


print("Forecasted Data Summary:")
print(forecast_summary)


plt.plot(data['Close'], label='Actual')
plt.plot(data.index, results_full.fittedvalues, label='Fitted', color='green')
plt.plot(forecast_values.index, forecast_values, label='Forecast', color='orange')
plt.title('ARIMA Model: Actual, Fitted, and Forecasted Values for NIFTY')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.show()
