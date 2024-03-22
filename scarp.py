import pandas as pd
import time
from selenium.webdriver.common.keys import Keys#untuk tekan tekan
from selenium.webdriver.common.by import By#untuk nyari element
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import selenium
import json
from datetime import datetime,timedelta
import pandas as pd
from undetected_chromedriver import Chrome, ChromeOptions
import numpy as np
import os

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

def scrap_tambahan():
    stock_code=pd.read_csv("data/processed/clean_database.csv")["StockCode"].unique()
    #buat fungsi iteratif
    start_date=pd.read_csv("data/processed/data_tambahan.csv")["Date"].max()
    now = datetime.now()
    one_day_before = now - timedelta(days=1)
    end_date = one_day_before.strftime("%Y-%m-%d")
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

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
        # Menetapkan ukuran jendela
        options.add_argument('window-size=1920x1080')
        # Mengganti user-agent untuk menghindari deteksi sebagai bot
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3')
        undetect = selenium.webdriver.Chrome(options=options)
        
        for i in formatted_dates:
            url = f"https://www.idx.co.id/primary/TradingSummary/GetStockSummary?length=9999&start=0&date={i}"
            undetect.get(url)
            WebDriverWait(undetect, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body > pre")))
            time.sleep(0.2)
            
            page_source = undetect.page_source
            if 'recordsTotal":0' in page_source:
                # Assuming change_date_format is a function you've defined elsewhere
                df = pd.DataFrame({"Date": [change_date_format(i) for _ in range(len(stock_code))],
                                "StockCode": list(stock_code),
                                "Close": ["unk" for _ in stock_code]})
                tmp = pd.concat([tmp, df], ignore_index=True)
            else:
                data = json.loads(undetect.find_element(By.TAG_NAME, 'pre').text)
                df = pd.DataFrame(data["data"])
                # Ensure the column names here match those in the JSON structure
                df = df[["Date", "StockCode", "Close"]]
                tmp = pd.concat([tmp, df], ignore_index=True)

    finally:
        time.sleep(2)
        undetect.quit()
        tmp=preprocess_data(tmp)
        return tmp


df_tambahan=scrap_tambahan()
print(df_tambahan)
#delete data_tambahan.csv
df_tambahan=pd.concat([df_tambahan,pd.read_csv("data/processed/data_tambahan.csv")],ignore_index=True)
# Delete the existing data_tambahan.csv file
df_tambahan.drop_duplicates(inplace=True)
os.remove("data/processed/data_tambahan.csv")
# Save the new df_tambahan DataFrame to data_tambahan.csv
df_tambahan.to_csv("data/processed/data_tambahan.csv", index=False)
