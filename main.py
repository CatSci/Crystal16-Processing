import streamlit as st
import pandas as pd
from utils import get_block_data, get_clear_and_cloud, convert_datatype, get_experiments_details, get_reactor_info, plot_reactor
import matplotlib.pyplot as plt
import re
import io 
import matplotlib
matplotlib.use('Agg')

st.header("Crystal Data Processing")

uploaded_file = st.file_uploader("Choose a file")


# Define a regex pattern to identify values like "2MeTHF:Hept(20:80)"
pattern = r"\b\d+\w+:\w+\(\d+:\d+\)\b"

# Function to replace matching values with NaN
def replace_values(match):
    return pd.NA

def download_btn(binary_image):
    file_name = "CrystalData"
    if file_name:
        file_name = str(file_name) + ".png"
        plt.savefig(file_name)
        st.download_button(
                    label="Download image",
                    data=binary_image,
                    file_name=file_name,
                    mime="image/png"
                )

def read_file(uploaded_file):
    # Read CSV file as a text file
    lines = uploaded_file.getvalue().decode('utf-8').splitlines()

    # Process the lines as needed
    # For example, split lines by delimiter
    data = [line.strip().split(',') for line in lines]
    df = pd.DataFrame(data)
    df.columns = df.iloc[0]
    df = df.iloc[1:]
    # Replace None values in the header with unique temporary column names
    df.columns = ['temp' + str(i) if col is None or col == '' else col for i, col in enumerate(df.columns)]

    return df

if st.button("Plot Graph"):
    with st.spinner("Please wait...."):
        if uploaded_file is not None:

            df = read_file(uploaded_file)
            block_data = get_block_data(dataframe= df)

            block_data = convert_datatype(dataframe= block_data)
            # st.write(block_data.dtypes.to_dict())

            clear, cloud = get_clear_and_cloud(dataframe=block_data)


            reactor_dataframe = get_experiments_details(dataframe= df)
            reactor_info = get_reactor_info(extracted_dataframe= reactor_dataframe)
            # st.write(reactor_info)
            
            
        
            plot = plot_reactor(final_dataframe= block_data, clear_dataframe= clear, cloud_dataframe= cloud)
            st.pyplot(plot)

            plot_binary = io.BytesIO()
            plt.savefig(plot_binary, format='png')
            plot_binary.seek(0)  # Move the stream pointer to the beginning
            # Close the plot to release resources
            
            plt.close(plot)
            download_btn(binary_image= plot_binary)
    