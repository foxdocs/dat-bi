import streamlit as st
import pandas as pd
import numpy as np
import random
import webbrowser
import io
from io import StringIO, BytesIO

from urllib.error import URLError
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio
import kaleido

import streamlit.components.v1 as components
from streamlit.components.v1 import html

import sweetviz as sv


import sys, os
import platform
sys.path.append('../')

st.set_page_config(page_title="Operations with Tabular Data", page_icon="📊")

st.title("Operations with Tabular Data")
st.sidebar.header("Tabular Data", divider='rainbow')
st.write(
            """This demo case illustrates ingestion and multiple visualisations of data that originates from tables, such as Excel or relational database tables. The data is loaded in a data frame structure and visualised by various diagrams."""
)


# Read the selected file
def readTabData(tab):
        df = pd.read_csv(tab)      
        st.dataframe(df, use_container_width=True)
        return df
    

# Use the analysis function from sweetviz module to create a 'Dataframe Report' object
def eda(df):
    my_report = sv.analyze([df,'EDA'])
    my_report.show_html('./html/eda.html', open_browser=False)     
    # components.iframe(src='http://localhost:3001/eda.html', width=1100, height=1200, scrolling=True)                
    return my_report


# Prepare data for vizualisation
def viz1(df):
        st.header('Data Visualisation 1')
        st.subheader('Grouping by a Nominal Attribute')
        a = st.selectbox('**Select the nominal attribute, A**', df.columns)
        b = st.selectbox('**Select first measure, B**', df.columns)
        c = st.selectbox('**Select second measure, C**', df.columns)     
        return a, b, c

    
# Prepare data for second vizualisation    
def viz2(df):
        st.header('Data Visualisation 2')
        st.subheader('Dimensions and Measures')
        x = st.selectbox('**Select the first dimension, X**', df.columns)
        z = st.selectbox('**Select the second dimension, Z**', df.columns)
        y = st.selectbox('**Select the measure, Y**', df.columns)     
        return x, y, z

    
# Design the visualisation
def charts():
        
            tab1, tab2, tab3, tab4 = st.tabs(['Bar Chart', 'Line Chart', "2D Scatter Plot", "3D Scatter Plot"])
            with tab1: # bar chart    
                st.bar_chart(df, x=a, y=[b, c], color=['#FF0000', '#0000FF'])  
                
            with tab2: # line chart
                st.line_chart(df, x=x, y=[y, z], color=["#FF0000", "#0000FF"])

            
            with tab3: # 2D scatter plot
                import altair as alt
                #st.scatter_chart(df, x=x, y=[y, z], size=z)
                ch = (alt.Chart(df).mark_circle().encode(x=x, y=y, size=z, color=z, tooltip=[x, y, z]))
                st.altair_chart(ch, use_container_width=True)                                
            
            with tab4: # 3D scatter plot 
                pio.templates.default = 'plotly'
                fig2 = px.scatter_3d(df, x=x, y=z, z=y, size=y, color=x)
                st.plotly_chart(fig2, theme='streamlit', use_container_width=True)            
                fig2.write_image("./media/MDskat.png")
    
    
# Main 
tab = ''
# tab = '../data/shopping-data.csv'

try:    
    tab = st.file_uploader("Choose a file with tab data (e.g. MS Excel of file of type .csv)")
    if tab is not None:
        df = readTabData(tab)
        #st.dataframe(df, use_container_width=True)
except:
    pass   
st.success("👆 Select the attributes of interest")
    
eda(df)
a, b, c = viz1(df)
x, y, z = viz2(df)

if st.button(":blue[EDA]"):
    # st.write(os.getcwd())
    with open("html/eda.html", "r", encoding='utf-8') as f:
        html_data = f.read() 
        components.html(html_data, width=1600, height=1600, scrolling=True)  

if st.button(":green[Explore]"):
            st.subheader("Explore the Data in Diagrams")
            st.write('Click on tabs to explore')
            container = st.container()
            charts()            
                  
 

        

    

