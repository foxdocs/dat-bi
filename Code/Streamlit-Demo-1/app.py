# Design Home Page
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from streamlit_option_menu import option_menu

import json
import requests
import pandas as pd
import numpy as np


from io import StringIO
from PIL import Image
logo = Image.open('./media/logo.png')

st.set_page_config(
    page_title="Streamlit BI Demo",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:tdi@cphbusiness.dk',
        'About': "https://docs.streamlit.io"
    }
)

st.sidebar.header("Try Me!", divider='rainbow')
# st.sidebar.success("Select a demo case from above")
st.image(logo, width=200)

banner = """
    <body style="background-color:yellow;">
            <div style="background-color:#385c7f ;padding:10px">
                <h2 style="color:white;text-align:center;">Streamlit BI Demo App</h2>
            </div>
    </body>
    """

st.markdown(banner, unsafe_allow_html=True)


st.markdown(
    """
    ###
        
    👈 :green[Select a demo case from the sidebar to experience some of what Streamlit can do for BI!]
    
    ### To learn more
    - Check out [Streamlit Documentation](https://docs.streamlit.io)
    - Contact me by [email](mailto://tdi@cphbusiness.dk)
"""
)






