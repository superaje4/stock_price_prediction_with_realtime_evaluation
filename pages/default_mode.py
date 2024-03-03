import streamlit as st
from st_pages import add_page_title,hide_pages

hide_pages(["Default Forcast"])
hide_pages(["Tunned Forcast"])
add_page_title()

# Menetapkan judul halaman
st.title('Default Mode Explanation')

# Menjelaskan tentang Default Mode
st.write("""
In the Default Mode, our forecasting model operates using real-time data up to the current moment. This approach 
ensures that the predictions are based on the most recent market information available, providing users with insights 
that are as current as possible.
""")

# Menjelaskan kelebihan
st.subheader('Advantages')
st.write("""
- **Real-time Data Utilization:** The model leverages the latest data available in real-time, ensuring that the 
forecasts are grounded in the most current market conditions.
""")

# Menjelaskan kekurangan
st.subheader('Disadvantages')
st.write("""
- **Non-Tuned Model:** The predictions are generated using a default model that has been pre-designed. This means 
that while the model provides quick and current insights, it may not be as refined or optimized as a model that 
has been tuned specifically for the current market conditions.
""")

# Opsional: Tambahkan lebih banyak informasi atau navigasi


st.write("Explore other features of our application to see how we can further assist you in making informed investment decisions.")
st.page_link("pages/forecasting_result_default.py", label="Forcast Now", icon="📈")

