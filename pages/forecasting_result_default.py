import streamlit as st
from st_pages import hide_pages
import pandas as pd
import time
from selenium.webdriver.common.keys import Keys#untuk tekan tekan
from selenium.webdriver.common.by import By#untuk nyari element
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import selenium
import bs4
from datetime import datetime,timedelta
import requests
import numpy as np
import json
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import r2_score
from keras.callbacks import Callback
from itertools import cycle
import plotly.express as px
import plotly.graph_objects as go
import sqlite3

#hide pages
hide_pages(["Default Forcast"])
hide_pages(["Tunned Forcast"])

#buat fungsi preprocessing
@st.cache_data
def preprocess_data(df):
    df["Date"] = [i.strip()[:10] for i in df["Date"]]
    df["Date"] = pd.to_datetime(df["Date"])

    df["Close"] = df["Close"].astype(str)
    df["Close"] = df["Close"].replace("unk", np.nan)
    df["Close"] = df["Close"].astype(float)
    
    # Pastikan bahwa kolom 'Date' adalah tipe datetime jika belum
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sortir DataFrame berdasarkan 'Date' untuk memastikan urutan yang benar
    df.sort_values(by=['Date'], inplace=True)
    
    # Pastikan bahwa kolom 'Date' adalah tipe datetime jika belum
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sortir DataFrame berdasarkan 'Date' untuk memastikan urutan yang benar
    df.sort_values(by=['Date'], inplace=True)
    
    def fill_group(group):
        # Set indeks grup menjadi 'Date' untuk interpolasi berbasis waktu
        group = group.set_index('Date')
        
        # Lakukan interpolasi berbasis waktu, lalu forward fill dan backward fill
        group['Close'] = group['Close'].interpolate(method='time').ffill().bfill()
        
        # Kembalikan indeks ke kolom sebelumnya untuk menghindari pengubahan struktur DataFrame asli
        group = group.reset_index()
        return group
    
    df = df.groupby('StockCode').apply(fill_group).reset_index(drop=True)
    return df
import pandas as pd
from datetime import datetime, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
import time
from selenium import webdriver
import os
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

from webdriver_manager.chrome import ChromeDriverManager


# Buat fungsi untuk melakukan scrape
@st.cache_data
def scrap_tambahan():
    stock_code = pd.read_csv("data/processed/clean_database.csv")["StockCode"].unique()
    start_date = '2024-03-02'
    now = datetime.now()
    one_day_before = now - timedelta(days=1)
    end_date = one_day_before.strftime("%Y-%m-%d")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Hilangkan jam, detik, dan milidetik
    formatted_dates = [date.strftime("%Y%m%d") for date in dates]  # Lebih simpel

    tmp = pd.DataFrame(columns=["Date", "StockCode", "Close"])
    

    options = Options()
    options.add_argument('--headless')  # Run Chrome in headless mode (without a visible browser window)
    options.add_argument('--disable-gpu')  # Disable GPU acceleration (can help with stability)
    # Menetapkan ukuran jendela
    options.add_argument('window-size=1920x1080')
   
    driver = webdriver.Chrome(executable_path='chromedriver-win64/chromedriver.exe')

    try:
        driver = webdriver.Chrome(ChromeDriverManager().install(), options=op)  # Pastikan path ChromeDriver sesuai
        for date in formatted_dates:
            try:
                url = f"https://www.idx.co.id/primary/TradingSummary/GetStockSummary?length=9999&start=0&date={date}"
                driver.get(url)
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body > pre")))
                time.sleep(0.2)
                
                data = json.loads(driver.find_element(By.TAG_NAME, 'pre').text)
                if data["recordsTotal"] == 0:
                    df = pd.DataFrame({
                        "Date": [date for _ in range(len(stock_code))],
                        "StockCode": list(stock_code),
                        "Close": ["unk" for _ in stock_code]
                    })
                else:
                    df = pd.DataFrame(data["data"])
                    df = df[["Date", "StockCode", "Close"]]
                tmp = pd.concat([tmp, df], ignore_index=True)
            except Exception as e:
                st.write(f"Error processing date {date}: {e}")
                continue
    except Exception as e:
        st.write(f"Error setting up WebDriver: {e}")
    finally:
        driver.quit()
        # tmp = preprocess_data(tmp)  # Pastikan Anda mendefinisikan fungsi ini
        return tmp

# @st.cache_data
# def scrap_tambahan():
#     stock_code=pd.read_csv("data/processed/clean_database.csv")["StockCode"].unique()
#     #buat fungsi iteratif
#     start_date = '2024-03-02'
#     now = datetime.now()
#     one_day_before = now - timedelta(days=1)
#     end_date = one_day_before.strftime("%Y-%m-%d")
#     dates = pd.date_range(start=start_date, end=end_date, freq='D')

#     #hilangkan jam detik dan milidetik
#     formatted_dates = [str(date).replace("-","") for date in dates]
#     formatted_dates = [date[:8] for date in formatted_dates]

#     #ubah formated dates ke format 2020-03-02
#     def change_date_format(date):
#         return f"{date[:4]}-{date[4:6]}-{date[6:]}"
    
    
#     try:
#         tmp = pd.DataFrame(columns=["Date", "StockCode", "Close"])
        
#         options = Options()
#         options.add_argument('--headless')  # Run Chrome in headless mode (without a visible browser window)
#         options.add_argument('--disable-gpu')  # Disable GPU acceleration (can help with stability)
#         # Menetapkan ukuran jendela
#         options.add_argument('window-size=1920x1080')
#         # Mengganti user-agent untuk menghindari deteksi sebagai bot
#         options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')
#         undetect = selenium.webdriver.Chrome(options=options)
        
#         for i in formatted_dates:
#             url = f"https://www.idx.co.id/primary/TradingSummary/GetStockSummary?length=9999&start=0&date={i}"
#             undetect.get(url)
#             WebDriverWait(undetect, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body > pre")))
#             time.sleep(0.2)
            
#             page_source = undetect.page_source
#             if 'recordsTotal":0' in page_source:
#                 # Assuming change_date_format is a function you've defined elsewhere
#                 df = pd.DataFrame({"Date": [change_date_format(i) for _ in range(len(stock_code))],
#                                 "StockCode": list(stock_code),
#                                 "Close": ["unk" for _ in stock_code]})
#                 tmp = pd.concat([tmp, df], ignore_index=True)
#             else:
#                 data = json.loads(undetect.find_element(By.TAG_NAME, 'pre').text)
#                 df = pd.DataFrame(data["data"])
#                 # Ensure the column names here match those in the JSON structure
#                 df = df[["Date", "StockCode", "Close"]]
#                 tmp = pd.concat([tmp, df], ignore_index=True)

#     finally:
#         time.sleep(2)
#         undetect.quit()
#         tmp=preprocess_data(tmp)
#         return tmp


@st.cache_data
def gabung_data(nama_perusahaan):
    df=pd.read_csv("data/processed/clean_database.csv")
    df1=scrap_tambahan()
    st.write(df1)
    df=pd.concat([df,df1],ignore_index=True)
    
    #spesifikasi namaperusahaan
    df_perusahaan=df[df["StockCode"]==nama_perusahaan]
    df_perusahaan["Date"]=df_perusahaan["Date"].astype(str)
    df_perusahaan=preprocess_data(df_perusahaan)
    return df_perusahaan
    

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def download_link(data_perusahaan):
    csv = convert_df(data_perusahaan)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='large_df.csv',
        mime='text/csv',
    )
    
# Inisialisasi session state untuk 'data_perusahaan'
st.session_state['data_perusahaan'] = None
st.title('Unlock Insights: Advanced Forecasting Models at Your Fingertips')

col1,col2=st.columns(2)
with col1:
    input=st.text_input('Write the IDX of the company')
    title=str(input)
    button_scrap = st.button('Scrap Data')
    if button_scrap:
        title=str(input)
        st.session_state['data_perusahaan'] = gabung_data(title)

with col2:
    if 'data_perusahaan' in st.session_state and st.session_state['data_perusahaan'] is not None:
        # Asumsi 'download_link' adalah fungsi yang Anda definisikan untuk mengunduh data
        download_link(st.session_state['data_perusahaan'])
        st.write(st.session_state['data_perusahaan'])
   
if title:
    plotdf = gabung_data(title)
    fig = px.line(plotdf, x='Date', y='Close', labels={'x': 'Date', 'y': 'Close Price'}, title=f"Stock Price {title}")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Tampilkan plot di Streamlit
    st.plotly_chart(fig)

    
    
    
# convert an array of values into a dataset matrix
@st.cache_data
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# Callback untuk update progress bar di Streamlit
class StreamlitProgressCallback(Callback):
    def __init__(self, max_epochs):
        self.max_epochs = max_epochs
        self.progress_bar = st.progress(0)

    def on_epoch_end(self, epoch, logs=None):
        # Update progress bar
        progress = (epoch + 1) / self.max_epochs
        self.progress_bar.progress(progress)

if "model_harian" not in st.session_state:
    st.session_state["model_harian"] = None
@st.cache_resource
def buat_model_harian():
    # normalisasi data
    data_model_harian=gabung_data(title).copy()
    scaler=MinMaxScaler(feature_range=(0,1))
    data_model_harian["norm"]=scaler.fit_transform(data_model_harian[["Close"]])

    # ambil kolom norm
    data_norm=data_model_harian[["norm"]].values.reshape(-1,1)

    # ubah ke tensor untuk lstm
    time_step = 15
    X_train, y_train = create_dataset(data_norm, time_step)
    
    # Buat model LSTM
    model_harian = Sequential()
    model_harian.add(LSTM(32, return_sequences=True, input_shape=(time_step, 1)))
    model_harian.add(LSTM(32, return_sequences=True))
    model_harian.add(LSTM(32))
    model_harian.add(Dense(1))
    model_harian.compile(loss='mean_squared_error', optimizer='adam')

    
    # Jumlah epoch untuk training
    max_epochs = 5

    # Buat instance callback Streamlit
    progress_callback = StreamlitProgressCallback(max_epochs)

    # Train model_harian dengan progress bar callback
    history = model_harian.fit(X_train, y_train, epochs=max_epochs, batch_size=5, verbose=1, callbacks=[progress_callback])

    # Lakukan prediksi dan cek metrik performa
    train_predict = model_harian.predict(X_train)
    # Transformasi kembali ke bentuk asli
    train_predict = scaler.inverse_transform(train_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    score = r2_score(original_ytrain, train_predict)
    
    st.write("r2 score: ",score)
            
    time_step = 15
    look_back = time_step
    trainPredictPlot = np.empty_like(data_model_harian['Close'])
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back] = train_predict.flatten()

    # Setup the cycle for legend names
    names = cycle(['Original close price', 'Train predicted close price'])

    plotdf = pd.DataFrame({
        'date': data_model_harian['Date'],
        'original_close': data_model_harian['Close'],
        'train_predicted_close': trainPredictPlot,
    })

    # Create a line plot with Plotly Express
    fig = px.line(plotdf, x='date', y=['original_close', 'train_predicted_close'],
                labels={'value': 'Stock price', 'date': 'Date'})
    
    # Update the layout of the plot
    fig.update_layout(
        title_text='Comparison between original Close price vs predicted Close price',
        plot_bgcolor='white',
        font_size=15,
        font_color='black',
        legend_title_text='Price'
    )
    
    # Update names of the traces
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    
    # Remove the gridlines from the plot
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    st.session_state["model_harian"] = model_harian

#untuk bulanan
if "model_bulanan" not in st.session_state:
    st.session_state["model_bulanan"] = None
@st.cache_resource
def buat_model_bulanan():
    # normalisasi data
    data_model_bulanan=gabung_data(title).copy()
    data_model_bulanan["Close"]=data_model_bulanan["Close"].astype(float)
    data_model_bulanan["Date"]=data_model_bulanan["Date"].astype(str)
    data_model_bulanan["Date"]=[i[:7] for i in data_model_bulanan["Date"]]
    data_model_bulanan=data_model_bulanan.groupby("Date")["Close"].mean().reset_index()
    
    scaler=MinMaxScaler(feature_range=(0,1))
    data_model_bulanan["norm"]=scaler.fit_transform(data_model_bulanan[["Close"]])

    # ambil kolom norm
    data_norm=data_model_bulanan[["norm"]].values.reshape(-1,1)

    # ubah ke tensor untuk lstm
    time_step = 7
    X_train, y_train = create_dataset(data_norm, time_step)

    #reshape input to be [samples, time steps, features] which is required for LSTM
    model_bulanan=Sequential()
    model_bulanan.add(LSTM(32,return_sequences=True,input_shape=(time_step,1)))
    model_bulanan.add(LSTM(32,return_sequences=True))
    model_bulanan.add(LSTM(32))
    model_bulanan.add(Dense(1))
    model_bulanan.compile(loss='mean_squared_error',optimizer='adam')
    
    
    # Jumlah epoch untuk training
    max_epochs = 20

    # Buat instance callback Streamlit
    progress_callback = StreamlitProgressCallback(max_epochs)
    
    model_bulanan.fit(X_train,y_train,epochs=max_epochs,batch_size=5, verbose=1, callbacks=[progress_callback])
    
    ### Lets Do the prediction and check performance metrics
    train_predict=model_bulanan.predict(X_train)
    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    score=r2_score(original_ytrain, train_predict)
    st.write("r2 score: ",score)
     
    time_step = 7
    look_back = time_step
    trainPredictPlot =  np.empty_like(data_model_bulanan['Close'])
    trainPredictPlot[:] = np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back] = train_predict.flatten()


    # Setup the cycle for legend names
    names = cycle(['Original close price', 'Train predicted close price'])

    plotdf = pd.DataFrame({
        'date': data_model_bulanan['Date'],
        'original_close': data_model_bulanan['Close'],
        'train_predicted_close': trainPredictPlot,
    })

    # Create a line plot with Plotly Express
    fig = px.line(plotdf, x='date', y=['original_close', 'train_predicted_close'],
                labels={'value': 'Stock price', 'date': 'Date'})
    
    # Update the layout of the plot
    fig.update_layout(
        title_text='Comparison between original Close price vs predicted Close price',
        plot_bgcolor='white',
        font_size=15,
        font_color='black',
        legend_title_text='Price'
    )
    
    # Update names of the traces
    fig.for_each_trace(lambda t: t.update(name=next(names)))
    
    # Remove the gridlines from the plot
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    st.session_state["model_bulanan"] = model_bulanan
    
col1,col2=st.columns(2)
with col1:
    check_model=st.checkbox("Build Daily Model",key="harian")
    if check_model:
        buat_model_harian()
            

with col2:
    check_model_2=st.checkbox("Build Monthly Model",key="bulanan")
    if check_model_2:
        buat_model_bulanan()
        

def buat_model_harian(nhari:int):
    df=gabung_data(title).copy()
    scaler=MinMaxScaler(feature_range=(0,1))
    df["norm"]=scaler.fit_transform(df[["Close"]])

    # ambil kolom norm
    data_norm=df[["norm"]].values.reshape(-1,1)

    # ubah ke tensor untuk lstm
    time_step = 15
    X_train, y_train = create_dataset(data_norm, time_step)
    
    # Lakukan prediksi dan cek metrik performa
    train_predict = st.session_state["model_harian"].predict(X_train)
    # Transformasi kembali ke bentuk asli
    train_predict = scaler.inverse_transform(train_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1, 1))
    score = r2_score(original_ytrain, train_predict)

    x_input=df[["norm"]][len(df)-time_step:].values.reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    #prediksi sebanyak data baru
    from numpy import array
    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = nhari
    while(i<pred_days):
        
        if(len(temp_input)>time_step):
            
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1,-1)

            yhat = st.session_state["model_harian"].predict(x_input)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
        
            lst_output.extend(yhat.tolist())
            i=i+1
            
        else:
            
            x_input = x_input.reshape((1, n_steps))
            yhat = st.session_state["model_harian"].predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            
            lst_output.extend(yhat.tolist())
            i=i+1
            
    # Konversi prediksi menjadi format yang sesuai dan menambahkannya ke DataFrame
    lst_output = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    #masukan hasil ke csv
     #buat tanggal dari one days hingga nhari
    one_day_start = datetime.now()
    date_rng=pd.date_range(start=one_day_start, end=one_day_start + timedelta(days=nhari), freq='B')
    if len(date_rng)!=nhari:
        date_rng=pd.date_range(start=one_day_start, end=one_day_start + timedelta(days=nhari+1), freq='B')
        if len(date_rng)!=nhari:
            date_rng=pd.date_range(start=one_day_start, end=one_day_start + timedelta(days=nhari-1), freq='B')
            if len(date_rng)!=nhari:
                date_rng=pd.date_range(start=one_day_start, end=one_day_start + timedelta(days=nhari+2), freq='B')
                
    date_rng = [date.strftime('%Y-%m-%d') for date in date_rng]
    data_baru=pd.DataFrame()
    data_baru["tanggal"]=date_rng
    data_baru['prediksi']=lst_output
    data_baru["accuracy"]=[score**(i+1) for i in range(data_baru.shape[0])]

    return data_baru, score


def buat_model_bulanan(nbulan:int):
    data_model_bulanan=gabung_data(title).copy()
    data_model_bulanan["Close"]=data_model_bulanan["Close"].astype(float)
    data_model_bulanan["Date"]=data_model_bulanan["Date"].astype(str)
    data_model_bulanan["Date"]=[i[:7] for i in data_model_bulanan["Date"]]
    df=data_model_bulanan.copy()
    data_model_bulanan=data_model_bulanan.groupby("Date")["Close"].mean().reset_index()

    scaler=MinMaxScaler(feature_range=(0,1))
    data_model_bulanan["norm"]=scaler.fit_transform(data_model_bulanan[["Close"]])

    # ambil kolom norm
    data_norm=data_model_bulanan[["norm"]].values.reshape(-1,1)

    # ubah ke tensor untuk lstm
    time_step = 7
    X_train, y_train = create_dataset(data_norm, time_step)

    ### Lets Do the prediction and check performance metrics
    train_predict=st.session_state["model_bulanan"].predict(X_train)
    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict)
    original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
    score=r2_score(original_ytrain, train_predict)

    x_input=data_model_bulanan[["norm"]][len(data_model_bulanan)-time_step:].values.reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    #prediksi sebanyak data baru
    from numpy import array

    lst_output=[]
    n_steps=time_step
    i=0
    pred_days = nbulan
    while(i<pred_days):
        if(len(temp_input)>time_step):
            
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input = x_input.reshape(1,-1)

            yhat = st.session_state["model_bulanan"].predict(x_input)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)

            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps))
            yhat = st.session_state["model_bulanan"].predict(x_input)
            temp_input.extend(yhat[0].tolist())
            
            lst_output.extend(yhat.tolist())
            i=i+1
        
       
            
    # Konversi prediksi menjadi format yang sesuai dan menambahkannya ke DataFrame
    lst_output = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(-1).tolist()
    #masukan hasil ke csv

    now = datetime.now()
    one_day_before = now
    one_day_until=  now + timedelta(days=30*nbulan-1)
    date_rng=pd.date_range(start=one_day_before, end=one_day_until, freq='B')
    

    data_baru=pd.DataFrame(date_rng, columns=['tanggal'])
    data_baru["tanggal"]=pd.to_datetime(data_baru["tanggal"])
    data_baru["prediksi"]=np.nan
    data_baru["tanggal"]=data_baru["tanggal"].astype(str)
    data_baru["tanggal"]=[i[:7] for i in data_baru["tanggal"]]
    data_baru=data_baru.groupby("tanggal")["prediksi"].mean().reset_index()
    #fillna dengan lst_output
    if len(lst_output)==len(data_baru):
        data_baru["prediksi"]=lst_output
    elif len(lst_output)<len(data_baru):
            data_baru=data_baru.iloc[0:-1,:]
            data_baru['prediksi']=lst_output
    else:
        now = datetime.now()
        one_day_before = now
        one_day_until=  now + timedelta(days=30*nbulan)
        date_rng=pd.date_range(start=one_day_before, end=one_day_until, freq='B')
        data_baru=pd.DataFrame(date_rng, columns=['tanggal'])
        data_baru["tanggal"]=pd.to_datetime(data_baru["tanggal"])
        data_baru["prediksi"]=np.nan
        data_baru["tanggal"]=data_baru["tanggal"].astype(str)
        data_baru["tanggal"]=[i[:7] for i in data_baru["tanggal"]]
        data_baru=data_baru.groupby("tanggal")["prediksi"].mean().reset_index()
        data_baru['prediksi']=lst_output  
    data_baru["accuracy"]=[score**(i+1) for i in range(data_baru.shape[0])]
    return data_baru,score


col1,col2=st.columns(2)
with col1:
    if check_model:
        st.write("Daily model is a model that is built based on daily stock price data. This model is suitable for short-term investment analysis.")
        nhari = st.text_input("What days would you want to predict until", key="hari")
        if nhari:
            nhari=int(nhari)
            st.write(f'Your predicting days {nhari} ahead:')
            data_baru,score=buat_model_harian(nhari)
            st.write(data_baru)
            st.write("r2 score: ",score)
            # buat plotly untuk data baru
            names = cycle(['Original harga price','Predicted harga price'])
            plotdf = pd.DataFrame({'Date': data_baru['tanggal'],                   
                                'predicted_close': data_baru['prediksi']})

            fig = px.line(plotdf,x=plotdf['Date'], y=plotdf['predicted_close'],
                            labels={'value':'Stock price','date': 'Date'})
            fig.update_layout(title_text=f'prediction of stock price in {nhari} days ahead',
                            plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
            fig.for_each_trace(lambda t:  t.update(name = next(names)))

            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig)
            
            with sqlite3.connect("data/database/stock.db") as con:
                cursor=con.cursor()
                data_baru.to_sql("forcasting_result_default_harian", con, if_exists="append", index=False)
                con.commit()
with col2:
    if check_model_2:
        st.write("Monthly model is a model that is built based on daily stock price data. This model is suitable for long-term investment analysis.")
        nbulan = st.text_input("What months would you want to predict until", key="bulan")
        if nbulan:
            nbulan=int(nbulan)
            st.write(f'Your predicting months {nbulan} ahead:')
            data_baru,score=buat_model_bulanan(nbulan)
            st.write(data_baru)
            st.write("r2 score: ",score)
            # buat plotly untuk data baru
            names = cycle(['Original harga price','Predicted harga price'])
            plotdf = pd.DataFrame({'Date': data_baru['tanggal'],                   
                                'predicted_close': data_baru['prediksi']})

            fig = px.line(plotdf,x=plotdf['Date'], y=plotdf['predicted_close'],
                            labels={'value':'Stock price','date': 'Date'})
            fig.update_layout(title_text=f'prediction of stock price in {nbulan} months ahead',
                            plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
            fig.for_each_trace(lambda t:  t.update(name = next(names)))

            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)
            st.plotly_chart(fig)
            
            with sqlite3.connect("data/database/stock.db") as con:
                cursor=con.cursor()
                data_baru.to_sql("forcasting_result_default_bulanan", con, if_exists="append", index=False)
                con.commit()
                