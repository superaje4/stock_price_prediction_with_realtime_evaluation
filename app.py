#buat dashboard untuk tampilan awal web forcasting dengan stream lit
import streamlit as st
import pandas as pd
from st_pages import Page, add_page_title, show_pages, Section, hide_pages


hide_pages(["Default Forcast"])
hide_pages(["Tunned Forcast"])

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('data/processed/data_model.csv')
    return data

data = load_data()

# Set title
st.title('Stock Forecasting For Indonesian Market')

st.write('''Welcome to the Indonesian Company Forecasting Application!
In a world filled with economic uncertainty, having insight into the future is key to making the right investment decisions. This application is designed to provide you with predictive analysis regarding the performance of companies listed on the Indonesia Stock Exchange, using the latest data and leading forecasting technology.
''')

st.markdown("""With this application, you can:

* Access stock performance predictions from various Indonesian companies.
* Understand market trends with data updated in real-time.
* Make better investment decisions with accurate and reliable information.
""")
st.write("Start exploring our features and discover valuable insights for your investments!")


import streamlit as st

# Displaying the Tutorial header on the main page
st.header('Tutorial')

# Adding text and a link
st.write("""
To select a company for forecasting, you need to grab the company IDX, you can visit the Indonesia Stock Exchange website and explore the listed company profiles. This information will assist you in making decisions about which company's stock data you wish to analyze further.
""")

st.markdown("""
Visit the following link to view the listed company profiles on IDX: [Listed Company Profiles IDX](https://www.idx.co.id/id/perusahaan-tercatat/profil-perusahaan-tercatat/)
""", unsafe_allow_html=True)

# Placeholder for the image
st.write("The tutorial image display will be shown here.")
# Once you have the image, you can add it using st.image() like the example below:
st.image("images/tutorial.png", caption="How to Select Companies on IDX")
st.write("This is an example of how to select companies on IDX.")
st.write("You can also type the company name in any browser and search for the ")




tab1, tab2, tab3 = st.tabs(["About Us", "Data Source", "Extras"])

with tab1:
    st.header("About Us")
    st.write("""
    Our application is designed to provide you with predictive analysis regarding the performance of companies listed on the Indonesia Stock Exchange, using the latest data and leading forecasting technology. We strive to deliver accurate and trustworthy stock performance predictions, so you can make better investment decisions with accurate and reliable information.
    """)

with tab2:
    st.header("Data Source")
    st.write("""
    Our application uses data from the official Indonesia Stock Exchange (BEI), which is updated in real-time. We use this data to provide you with the most accurate and up-to-date stock performance predictions, so you can make informed investment decisions. Our data is sourced from reliable and trustworthy sources, so you can trust the information we provide.
    """)

with tab3:
    st.header("Extras")
    st.write("""
    Why did the stock market investor cross the road?
    To buy shares on the other side where the grass is always greener!
    """)


show_pages([
    Page("app.py", "Home", "🏠"),
    Section("Choose Mode"),
    Page("pages/default_mode.py", "Default Mode", "🧰"),
    Page("pages/forecasting_result_default.py", "Default Forcast"),
    Page("pages/tunned_mode.py", "Tunned Mode", "📖"),
    Page("pages/forecasting_result_tunned.py", "Tunned Forcast")])


add_page_title()

