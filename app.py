# app.py
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

def generate_data(periods=48):
    dates = pd.date_range(start='2020-01-01', periods=periods, freq='M')
    sales_volume = np.random.randint(1000, 1800, size=periods) * (1 + 0.01 * np.sin(np.linspace(0, 6 * np.pi, periods)))
    logistics_cost = np.random.uniform(40000, 60000, size=periods)
    production_capacity = np.random.randint(50, 100, size=periods)
    competitor_price = np.random.uniform(200, 280, size=periods)
    inflation_rate = np.random.uniform(1, 2, size=periods)

    data = pd.DataFrame({
        'sales_volume': sales_volume,
        'logistics_cost': logistics_cost,
        'production_capacity': production_capacity,
        'competitor_price': competitor_price,
        'inflation_rate': inflation_rate
    }, index=dates).round(2)
    return data

def run_model():
    data = generate_data()
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    endog_train = train_data['sales_volume']
    exog_train = train_data[['logistics_cost', 'production_capacity', 'competitor_price', 'inflation_rate']]
    exog_test = test_data[['logistics_cost', 'production_capacity', 'competitor_price', 'inflation_rate']]

    model = SARIMAX(endog=endog_train, exog=exog_train, order=(1,1,1), seasonal_order=(1,1,1,12))
    results = model.fit(disp=False)

    forecast = results.get_forecast(steps=len(test_data), exog=exog_test)
    predictions = forecast.predicted_mean
    true_values = test_data['sales_volume']

    mape = mean_absolute_percentage_error(true_values, predictions) * 100
    return test_data.index, true_values, predictions, mape

st.title("🔮 Прогноз объёма продаж")
st.write("Нажмите кнопку ниже, чтобы запустить модель.")

if st.button("🚀 Запустить прогноз"):
    with st.spinner("Обучение модели..."):
        dates, y_true, y_pred, mape = run_model()

    st.success(f"✅ MAPE: {mape:.2f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, y_true, label="Фактические значения")
    ax.plot(dates, y_pred, label="Прогноз", color="red")
    ax.legend()
    st.pyplot(fig)