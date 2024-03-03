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
import pandas as pd
import requests
from undetected_chromedriver import Chrome, ChromeOptions
import numpy as np



#buat fungsi iteratif
start_date = '2020-01-01'
now = datetime.now()
one_day_before = now - timedelta(days=1)
end_date = one_day_before.strftime("%Y-%m-%d")
dates = pd.date_range(start=start_date, end=end_date)
dates


#hilangkan jam detik dan milidetik
formatted_dates = [str(date).replace("-","") for date in dates]
formatted_dates = [date[:8] for date in formatted_dates]
formatted_dates

#ubah formated dates ke format 2020-03-02
def change_date_format(date):
    return f"{date[:4]}-{date[4:6]}-{date[6:]}"




#buat fungsi iteratif
def scrap_database():
    start_date = '2020-01-01'
    now = datetime.now()
    one_day_before = now - timedelta(days=1)
    end_date = one_day_before.strftime("%Y-%m-%d")
    dates = pd.date_range(start=start_date, end=end_date)
    dates


    #hilangkan jam detik dan milidetik
    formatted_dates = [str(date).replace("-","") for date in dates]
    formatted_dates = [date[:8] for date in formatted_dates]
    formatted_dates
    try:
        tmp=pd.DataFrame(columns=["Date","StockCode","Close"])
        
        #headless
        # options = ChromeOptions()
        # options.add_argument('--headless')  # Run Chrome in headless mode (without a visible browser window)
        # options.add_argument('--disable-gpu')  # Disable GPU acceleration (can help with stability)
        undetect = Chrome()
        
        for i in formatted_dates:
            url=f"https://www.idx.co.id/primary/TradingSummary/GetStockSummary?length=9999&start=0&date={i}"
            undetect.get(url)
            WebDriverWait(undetect, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "body > pre")))
            
            #buat random sleep
            time.sleep(0.5)
            if undetect.page_source == '<html><head><meta name="color-scheme" content="light dark"></head><body><pre style="word-wrap: break-word; white-space: pre-wrap;">{"draw":0,"recordsTotal":0,"recordsFiltered":0,"data":[]}</pre></body></html>':
                df=pd.DataFrame({"Date":[change_date_format(i) for j in range(len(set(tmp["StockCode"])))],"StockCode":[j for j in list(set(tmp["StockCode"]))],"Close":["unk" for j in range(len(set(tmp["StockCode"])))]})
                tmp=pd.concat([tmp,df],ignore_index=True)
            else:
                soup=bs4.BeautifulSoup(page_source,"html.parser")
                text=soup.find_all("body")[0].text
                data=json.loads(text)
                df=pd.DataFrame(data["data"])
                df=df[["Date","StockCode","Close"]]
                tmp=pd.concat([tmp,df],ignore_index=True)

        undetect.quit()
    except:
        undetect.quit()
        print("error")

    return tmp
#hasil dari fungsi scrap
df=pd.read_csv("data/raw/database.csv")

df1=df.copy()
df1["Date"]=[i[:10] for i in df1["Date"]]
df1["Date"]=pd.to_datetime(df1["Date"])
df1.sort_values("Date",inplace=True)    
df1.reset_index(drop=True,inplace=True)


df1["Close"]=df1["Close"].astype(str)
df1["Close"]=df1["Close"].replace("unk",np.nan)
df1["Close"]=df1["Close"].astype(float)

df1.set_index("Date",inplace=True)

#fillna dengan interpolasi
df1=df1.groupby("StockCode").apply(lambda x:x.interpolate(method="time"))
df1.reset_index(inplace=True,drop=True)
#fillna dengan ffil
df1=df1.groupby("StockCode").apply(lambda x:x.ffill())
df1.reset_index(inplace=True,drop=True)
#fillna dengan bfill
df1=df1.groupby("StockCode").apply(lambda x:x.bfill())
df1.reset_index(inplace=True,drop=True)

df1.to_csv("data/processed/clean_database.csv",index=False)

df1[df1["StockCode"]=="AALI"].Close.plot()

df1[df1["StockCode"]=="AALI"].isna().sum()


import pandas as pd
import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

def scrap_database_terbaru():
    try:
        tmp = pd.DataFrame(columns=["Date", "StockCode", "Close"])
        
        
        # Assuming the path to chromedriver is set in your PATH, otherwise specify the executable_path argument
        undetect = Chrome()
        
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
        undetect.quit()
        return tmp

database=scrap_database_terbaru(formatted_dates)

database.to_csv("data/raw/full_database.csv",index=False)

df=pd.read_csv("data/raw/full_database.csv")
df["Date"] = [i.strip()[:10] for i in df["Date"]]
df["Date"] = pd.to_datetime(df["Date"])

df["Close"] = df["Close"].astype(str)
df["Close"] = df["Close"].replace("unk", np.nan)
df["Close"] = df["Close"].astype(float)



def fillna_khusus(df):
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
    
    # Groupby 'StockCode' dan terapkan fungsi pengisian pada setiap grup
    df = df.groupby('StockCode').apply(fill_group).reset_index(drop=True)
    
    return df

df_no_na = fillna_khusus(df)


df["Close"].isna().sum()
df_no_na["Close"].isna().sum()

df_no_na.to_csv("data/processed/clean_database.csv",index=False)
#data yang sudah bersih di scrap hingga tanggal 2024-03-01
#maka untuk scrap lanjutan gunakan data mulai dari 2024-03-02

#buat fungsi preprocessing
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
    #buat fungsi iteratif
    start_date = '2024-03-02'
    now = datetime.now()
    one_day_before = now - timedelta(days=1)
    end_date = one_day_before.strftime("%Y-%m-%d")
    dates = pd.date_range(start=start_date, end=end_date)

    #hilangkan jam detik dan milidetik
    formatted_dates = [str(date).replace("-","") for date in dates]
    formatted_dates = [date[:8] for date in formatted_dates]
    formatted_dates

    #ubah formated dates ke format 2020-03-02
    def change_date_format(date):
        return f"{date[:4]}-{date[4:6]}-{date[6:]}"
    
    
    try:
        tmp = pd.DataFrame(columns=["Date", "StockCode", "Close"])
        
        
        # Assuming the path to chromedriver is set in your PATH, otherwise specify the executable_path argument
        undetect = Chrome()
        
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
        
        undetect.quit()
        tmp=preprocess_data(tmp)
        return tmp
    
    
def gabung_data(nama_perusahaan):
    df=pd.read_csv("data/processed/clean_database.csv")
    df1=scrap_tambahan()
    df=pd.concat([df,df1],ignore_index=True)
    
    #spesifikasi namaperusahaan
    df_perusahaan=df[df["StockCode"]==nama_perusahaan]
    
    return df_perusahaan
    


    
