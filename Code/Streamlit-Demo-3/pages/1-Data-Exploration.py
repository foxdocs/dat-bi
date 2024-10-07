import streamlit as st
import pandas as pd
import numpy as np
import random
import webbrowser
import io
from io import StringIO, BytesIO

import altair as alt
from urllib.error import URLError
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.io as pio

import streamlit.components.v1 as components
from streamlit.components.v1 import html

# import kaleido
from PIL import Image
logo = Image.open('./media/logo.png')

import sys, os
import platform
sys.path.append('../')

st.set_page_config(page_title="BI Playground", page_icon="📊")

st.image(logo, width=200)
st.title("BI Playground")
st.subheader("Operations with Tabular Data", divider='rainbow')
st.sidebar.header("Tabular Data")
st.write(
            """This demo case illustrates ingestion and multiple visualisations of data that originates from tables, such as Excel or relational database tables. The data is first loaded in a data frame structure, and later transfered into a graph, where it integrates with unstructured data from variety of other sources to form the domain of the business knowledge."""
)

st.success("👈 :green[Select a file using the file loading tool on the left]")


tab =''

def readTabData(tab):
        df = pd.read_csv(tab)      
        st.dataframe(df, use_container_width=True)
        st.write('Two of dataframe attributes (columns) will be used as (independent) dimensions for exploring the dynamics of a third (dependent) one, called meassure.')

        x = st.selectbox('**Select the first dimension, X**', df.columns)
        st.write(x)
        z = st.selectbox('**Select the second dimension, Z**', df.columns)
        st.write(z)
        y = st.selectbox('**Select the measure, Y**', df.columns)     
        st.write(y)

        # ddf = pd.read_csv('/Users/tdi/Documents/Holodeck/Holodeck/middle.csv')  
        df = df.astype({x: str})
        
        sy = df.groupby([x, z]).size().reset_index(name='count of (' + y + ')')
        # sy = df.groupby([x, z])[y].sum()
        
        st.write('Here is a new data frame, showing the two dimensions and the aggregated measure.')
        st.dataframe(sy, use_container_width=True)
        return sy, x, 'count of (' + y + ')', z

  
    
# Design the visualisation
def charts():
            chart1 = {
                "mark": {'type': 'point', 'tooltip': True},
                "encoding": {
                    "x": { "field": x, "type": "nominal"},
                    "y": {"field": y, "type": "quantitative" },
                    "color": {"field": z, "type": "nominal"},
                    "shape": {"field": z, "type": "nominal"},
                }
            }

            chart2 = {
                "mark": {'type': 'circle', 'tooltip': True},
                "encoding": {
                    "x": { "field": x,  "type": "nominal" },
                    "y": { "field": y,  "type": "quantitative" },
                    "color": {"field": z, "type": "nominal"},
                    "size": {"field": y, "type": "quantitative"},
                }
            }


            tab1, tab2, tab3, tab4, tab5 = st.tabs(["2D Scatter", "2D Scatter", 'Bar Chart', 'Sunburst Chart', '3D Scatter Plot'])
            with tab1:
                st.vega_lite_chart(sy, chart1, use_container_width=True)
            
            with tab2:
                st.vega_lite_chart(sy, chart2, theme=None, use_container_width=True)
            
            with tab3:
                st.bar_chart(sy, x=x, y=y, color=z)   
            
            with tab4:                
                fig1 = px.sunburst(sy, path=[ z, x, y], values=y)
                st.plotly_chart(fig1,  use_container_width=True, use_container_height=True) 
                fig1.write_html("./media/line-chart.html")   
              
            with tab5: # 3D scatter plot 
                pio.templates.default = 'streamlit'
                fig2 = px.scatter_3d(sy, x=z, y=x, z=y, size=y, color=x)
                st.plotly_chart(fig2, theme='streamlit', use_container_width=True)            
                pio.write_image(fig2, "./media/MDskat.svg")   
                

# print stored images
def images():
    graphselector = st.selectbox('**Select a feature you want to explore**', 
                (['Attrition shown per department',
                'Histogram of all features',
                'Age distribution',
                'Heatmap of correlations between features',
                'Years of employment in the company']),
    index=None,
    placeholder="select image")
    st.write(graphselector)
    
    graph = None
    
    match graphselector:
        case 'Attrition shown per department':
            graph = Image.open('./Media/Attrition shown in different department.png')
        case 'Histogram of all features':
            graph = Image.open('./Media/Histogram of all features.png')
        case 'Age distribution':
            graph = Image.open('./Media/Age distribution in the company.png')
        case 'Heatmap of correlations between features':
            graph = Image.open('./Media/Heatmap of correlation between all features.png')
        case 'Years of employment in the company':
            graph = Image.open('./Media/Number of years in the company.png')
    
    if graph is not None:
        st.image(graph)
    else:
        st.write('No')
    return graph  
        
try:    
    tab = st.sidebar.file_uploader("Choose a file with tab data (e.g. MS Excel of file of type .csv)")
    if tab is not None:
        sy,x,y,z = readTabData(tab)  
except:
    pass

st.subheader("Data Visualisation", divider='rainbow')
st.success("👆 Select the attributes of interest")  

if st.button(":green[Explore]"):
            st.subheader("Explore the Data in Diagrams")
            st.write('Click on tabs to explore')
            container = st.container()
            charts()

with st.expander(":green[Click here to see more diagrams]"):
    images()

# save fig
# import matplotlib.pyplot as pl
# fig = pl.hist(a,normed=0)
# pl.savefig("abc.png")

