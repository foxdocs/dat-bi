import streamlit as st
import pandas as pd
from PIL import Image

df = pd.read_csv('./Data/WA_Fn-UseC_-HR-Employee-Attrition.csv')

st.title("Data Exploration")
st.sidebar.header("Data Exploration", divider='rainbow')

st.write("Below you can take a look at what the data looks like after being anonymised.")

st.dataframe(df, use_container_width=True)

st.success("👇 Select a graph that you want to take a loot at below 👇")

graphselector = st.selectbox('**Select a graph you want to explore**', 
            ['Attrition shown per department',
            'Histogram of all features',
            'Age distribution',
            'Heatmap of correlations between features',
            'Years of employment in the company'])

graph = None

match graphselector:
    case 'Attrition shown per department':
        graph = Image.open('./Data/Exploration/Attrition shown in different department.png')
    case 'Histogram of all features':
        graph = Image.open('./Data/Exploration/Histogram of all features.png')
    case 'Age distribution':
        graph = Image.open('./Data/Exploration/Age distribution in the company.png')
    case 'Heatmap of correlations between features':
        graph = Image.open('./Data/Exploration/Heatmap of correlation between all features.png')
    case 'Years of employment in the company':
        graph = Image.open('./Data/Exploration/Numberr of years in the company.png')

if graph is not None:
    st.image(graph, use_column_width=True)