import streamlit as st
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
import numpy as np
from numpy import sqrt
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')


def main():
    st.sidebar.header('App Forecaster')
    st.sidebar.info("Created and designed by [Jonaben](https://www.linkedin.com/in/jonathan-ben-okah-7b507725b)")
    st.sidebar.info('Make sure your data is a time series data \
                    with just two columns including date')
    option = st.sidebar.selectbox('How do you want to get the data?', ['url', 'file'])
    if option == 'url':
        url = st.sidebar.text_input('Enter a url')
        if url:
            dataframe(url)
    else:
        file = st.sidebar.file_uploader('Choose a file', type=['csv', 'txt'])
        if file:
            dataframe(file)



def dataframe(df):
    st.header('App Forecaster')
    data = read_csv(df, header=0, parse_dates=True, index_col=0)
    to_do = st.radio('SELECT WHAT YOU WOULD LIKE TO DO WITH THE DATA', ['Visualize', 'Check for stationary', 'Forecast'])
    if to_do == 'Visualize':
        data_visualization(data)
    elif to_do == 'Check for stationary':
        stationary_test(data)
    else:
        forecast_data(data)



def data_visualization(data):
    button = st.button('Draw')
    if button:
        st.line_chart(data)



def stationary_test(data):
    res = testing(data)
    st.text(f'Augmented Dickey_fuller Statistical Test: {res[0]} \
           \np-values: {res[1]}')
    st.text('Critical values at different levels:')
    for k, v in res[4].items():
        st.text(f'{k}:{v}')
    if res[1] > 0.05:
        st.text('Your data is non-stationary and is being transformed \
               \nto a stationary time series data. ')
        if st.button('Check results'):
            data_transform(data)
    elif res[1] <= 0.05:
        st.text('Your data is stationary and is ready for training.')


def testing(df):
    return adfuller(df)


def data_transform(df):
    df_log = np.log(df.iloc[:, 0])
    df_diff = df_log.diff().dropna()
    res = testing(df)
    if res[1] < 0.05:
        st.line_chart(df_dff)
        st.write('1st order differencing')
    else:
        df_diff_2 = df_diff.diff().dropna()
        st.line_chart(df_diff_2)
        st.write('2nd order differencing')
        stationary_test(df_diff_2)

def forecast_data(df):
    st.text('...searching for the optimum parameter')
    optimum_para(df)
    st.text('Enter the parameter with the lowest RMSE')
    p = st.number_input('The p term')
    q = st.number_input('The q term')
    d = st.number_input('The d term')
    period = st.number_input('Enter the next period(s) you want to forecast', value=7)
    button = st.button('Forecast')
    if button:
        model_forecast(df, p, q, d, period)


def model_forecast(data, p, q, d, period):
    size = int(len(data) * .7)
    train, test = data[:size], data[size:]
    model = ARIMA(train.values, order=(p,q,d))
    model_fit = model.fit()
    output = model_fit.predict(start=len(train), end=len(train)+len(test)-1)
    error = sqrt(mean_squared_error(output, test))
    st.text(f'RMSE using {p,q,d}: {error}')
    st.text(f'Forecasting {period} future values')
    model_2 = ARIMA(data.values, order =(p,q,d)).fit()
    forecast = model_2.predict(start=len(data), end=len(data)+period, typ='levels')
    day = 1
    for i in forecast:
        st.text(f'Period {day}: {i}')
        day += 1



def optimum_para(df):
    p_values = [0, 1, 2]
    d_values = range(0, 3)
    q_values = range(0,3)
    size = int(len(df) * .7)
    train, test = df[:size], df[size:]
    for p in p_values:
        for q in q_values:
            for d in d_values:
                order = (p,q,d)
                model = ARIMA(train, order=order).fit()
                preds = model.predict(start=len(train), end=len(train) + len(test)-1)
                error = sqrt(mean_squared_error(test, preds))
                st.text(f'ARIMA {order} RMSE: {error}')



if __name__ == '__main__':
    main()
