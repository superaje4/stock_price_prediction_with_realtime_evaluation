#bantu saya buat fast api
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import requests
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
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import r2_score


app=FastAPI()

#buat api sedehana
@app.get('/')
def index():
    return {'message':'Selamat datang di API FastAPI Stock Forcasting Indonesia'}

#api scrappng
def scrap_tambahan():
    #buat fungsi iteratif
    start_date = '2024-03-02'
    now = datetime.now()
    one_day_before = now - timedelta(days=1)
    end_date = one_day_before.strftime("%Y-%m-%d")
    dates = pd.date_range(start=start_date, end=end_date)

    #hilangkan jam detik dan milidetik
    formatted_dates = [str(date).replace("-","") for date in dates]
    formatted_dates = [date[:8] for date in formatted_dates]


    #ubah formated dates ke format 2020-03-02
    def change_date_format(date):
        return f"{date[:4]}-{date[4:6]}-{date[6:]}"
    
    
    try:
        tmp = pd.DataFrame(columns=["Date", "StockCode", "Close"])
        
        
        options = Options()
        options.add_argument('--headless')  # Run Chrome in headless mode (without a visible browser window)
        options.add_argument('--disable-gpu')  # Disable GPU acceleration (can help with stability)
        undetect = selenium.webdriver.Chrome(options=options)
        
        for i in formatted_dates:
            url = f"https://www.idx.co.id/primary/TradingSummary/GetStockSummary?length=9999&start=0&date={i}"
            undetect.get(url)
            WebDriverWait(undetect, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body > pre")))
            time.sleep(0.2)
            
            page_source = undetect.page_source
            if 'recordsTotal":0' in page_source:
                # Assuming change_date_format is a function you've defined elsewhere
                df = pd.DataFrame({"Date": [change_date_format(i) for _ in range(len(tmp["StockCode"].unique()))],
                                "StockCode": list(tmp["StockCode"].unique()),
                                "Close": ["unk" for _ in tmp["StockCode"].unique()]})
                tmp = pd.concat([tmp, df], ignore_index=True)
            else:
                data = json.loads(undetect.find_element(By.TAG_NAME, 'pre').text)
                df = pd.DataFrame(data["data"])
                # Ensure the column names here match those in the JSON structure
                df = df[["Date", "StockCode", "Close"]]
                tmp = pd.concat([tmp, df], ignore_index=True)
    except:
        print("An error occurred")

    finally:
        time.sleep(2)
        undetect.quit()
        tmp=preprocess_data(tmp)
        return tmp
    
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
    
def gabung_data(nama_perusahaan):
    df=pd.read_csv("data/processed/clean_database.csv")
    df1=scrap_tambahan()
    df=pd.concat([df,df1],ignore_index=True)
    
    #spesifikasi namaperusahaan
    df_perusahaan=df[df["StockCode"]==nama_perusahaan]
    
    return df_perusahaan
#buat templete api untuk web scrapping

@app.get('/scraping')
def get_scraping(nama_perusahaan:str):
    response=gabung_data(nama_perusahaan)
    return response.to_json(orient="records")

#prediksi default
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)
#ambil data dari get_scraping

@app.post('/forcasting_default/harian')
def buat_model_harian(nama_perusahaan:str, nhari:int):
    df=gabung_data(nama_perusahaan)
    scaler=MinMaxScaler(feature_range=(0,1))
    df["norm"]=scaler.fit_transform(df[["Close"]])

    # ambil kolom norm
    data_norm=df[["norm"]].values.reshape(-1,1)

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


    # Train model_harian dengan progress bar callback
    model_harian.fit(X_train, y_train, epochs=max_epochs, batch_size=5)

    # Lakukan prediksi dan cek metrik performa
    train_predict = model_harian.predict(X_train)
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

            yhat = model_harian.predict(x_input)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
        
            lst_output.extend(yhat.tolist())
            i=i+1
            
        else:
            
            x_input = x_input.reshape((1, n_steps))
            yhat = model_harian.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            
            lst_output.extend(yhat.tolist())
            i=i+1
            
    # Konversi prediksi menjadi format yang sesuai dan menambahkannya ke DataFrame
    lst_output = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

    #masukan hasil ke csv
    now = datetime.now()
    one_day_before = now
    one_day_until=  now + timedelta(days=nhari-1)
    try:
        date_rng=pd.date_range(start=one_day_before, end=one_day_until, freq='B')

        data_baru=pd.DataFrame(date_rng, columns=['tanggal'])
        data_baru["tanggal"]=pd.to_datetime(data_baru["tanggal"])
        data_baru['prediksi']=lst_output
    except:
        date_rng=pd.date_range(start=one_day_before, end=now + timedelta(days=nhari), freq='B')

        data_baru=pd.DataFrame(date_rng, columns=['tanggal'])
        data_baru["tanggal"]=pd.to_datetime(data_baru["tanggal"])
        data_baru['prediksi']=lst_output

    #ubah data baru ke json
    data_baru=data_baru.to_json(orient="records")
    #tambahkan score ke databaru
    data_baru=json.loads(data_baru)
    data_baru[0]["score"]=score
    return data_baru


@app.post('/forcasting_default/bulanan')
def buat_model_bulanan(nama_perusahaan:str,nbulan:int):
    data_model_bulanan=gabung_data(nama_perusahaan)
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

    #reshape input to be [samples, time steps, features] which is required for LSTM
    model_bulanan=Sequential()
    model_bulanan.add(LSTM(32,return_sequences=True,input_shape=(time_step,1)))
    model_bulanan.add(LSTM(32,return_sequences=True))
    model_bulanan.add(LSTM(32))
    model_bulanan.add(Dense(1))
    model_bulanan.compile(loss='mean_squared_error',optimizer='adam')


    # Jumlah epoch untuk training
    max_epochs = 20


    model_bulanan.fit(X_train,y_train,epochs=max_epochs,batch_size=5)

    ### Lets Do the prediction and check performance metrics
    train_predict=model_bulanan.predict(X_train)
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

            yhat = model_bulanan.predict(x_input)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)

            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps))
            yhat = model_bulanan.predict(x_input)
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
                
    data_baru=data_baru.to_json(orient="records")
    # tambahkan score ke databaru
    data_baru=json.loads(data_baru)
    data_baru[0]["score"]=score
    return data_baru

# streamlit_app.py

import streamlit as st

# Create the SQL connection to pets_db as specified in your secrets file.
conn = st.connection('pets_db', type='sql')

# Insert some data with conn.session.
with conn.session as s:
    s.execute('CREATE TABLE IF NOT EXISTS pet_owners (person TEXT, pet TEXT);')
    s.execute('DELETE FROM pet_owners;')
    pet_owners = {'jerry': 'fish', 'barbara': 'cat', 'alex': 'puppy'}
    for k in pet_owners:
        s.execute(
            'INSERT INTO pet_owners (person, pet) VALUES (:owner, :pet);',
            params=dict(owner=k, pet=pet_owners[k])
        )
    s.commit()

# Query and display the data you inserted
pet_owners = conn.query('select * from pet_owners')
st.dataframe(pet_owners)