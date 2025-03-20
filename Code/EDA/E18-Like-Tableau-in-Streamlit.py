import pandas as pd
import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer

# Streamlit page
st.set_page_config(page_title="PyGWalker In Streamlit", layout="wide")
st.header("PyGWalker In Streamlit")

# Data
df = pd.read_csv("https://kanaries-app.s3.ap-northeast-1.amazonaws.com/public-datasets/bike_sharing_dc.csv")
pyg_app = StreamlitRenderer(df)


# Or diagram code copied from another Streamlit page
# vis_spec = 
# pyg_app = StreamlitRenderer(df, spec=vis_spec)

# Both
pyg_app.explorer()
