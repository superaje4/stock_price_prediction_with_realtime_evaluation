import streamlit as st
from st_pages import add_page_title,hide_pages

add_page_title()
hide_pages(["Default Forcast"])
hide_pages(["Tunned Forcast"])

# Menetapkan judul halaman
st.title('Tuned Model Explanation')

# Menjelaskan tentang Tuned Model
st.write("""
In the Tuned Model mode, our forecasting model has undergone a tuning process to optimize its performance. This approach 
ensures that the model operates at its best by using previously trained data. Additionally, real-time data obtained 
through scrapping is incorporated as an evaluation mechanism to assess the model's accuracy and reliability.
""")

# Menjelaskan kelebihan
st.subheader('Advantages')
st.write("""
- **Tuned Model:** The use of a model that has been carefully tuned means that it is expected to perform optimally, 
taking into account the nuances and complexities of the market data.
- **Reliable Evaluation Metrics:** With the model being evaluated against a scenario with truncated real-time data, 
the evaluation metrics provided are more reliable and reflective of the model's ability to predict under specific conditions.
""")

# Menjelaskan kekurangan
st.subheader('Disadvantages')
st.write("""
- **Limited Training Data:** As the model does not use the entirety of the data for training, there might be potential insights 
or patterns that are not captured by the model. This could affect the model's ability to generalize to new, unseen data.
""")

# Opsional: Tambahkan lebih banyak informasi atau navigasi

st.write("Explore the application further to understand how the Tuned Model can aid in your investment decisions and how we continuously strive")
st.page_link("pages/forecasting_result_tunned.py", label="Forcast Now", icon="📈")
