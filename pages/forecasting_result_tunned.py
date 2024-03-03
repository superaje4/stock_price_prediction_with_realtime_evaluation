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

@st.cache_data
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
    
@st.cache_data
def gabung_data(nama_perusahaan):
    df=pd.read_csv("data/processed/clean_database.csv")
    df1=scrap_tambahan()
    df=pd.concat([df,df1],ignore_index=True)
    
    #spesifikasi namaperusahaan
    df_perusahaan=df[df["StockCode"]==nama_perusahaan]
    
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
    title = st.text_input('Write the IDX of the company')
    scrape_button = st.button('Scrape data')
    if scrape_button:
        st.session_state['data_perusahaan'] = gabung_data(title)

with col2:
     if 'data_perusahaan' in st.session_state and st.session_state['data_perusahaan'] is not None:
        # Asumsi 'download_link' adalah fungsi yang Anda definisikan untuk mengunduh data
        download_link(st.session_state['data_perusahaan'])
        st.write(st.session_state['data_perusahaan'])
    
# Plot data
if 'data_perusahaan' in st.session_state and st.session_state['data_perusahaan'] is not None:
    plotdf = gabung_data(title)
    fig = px.line(plotdf, x='Date', y='Close', labels={'x': 'Date', 'y': 'Close Price'}, title=f"Stock Price {title}")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Tampilkan plot di Streamlit
    st.plotly_chart(fig)

