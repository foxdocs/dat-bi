import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

#@st.cache_data
def readData(tab):
    dff = pd.read_csv(tab)
    st.dataframe(dff, use_container_width=True) 

    st.success("👇 Select the columns with geospatial data of interest")

    name = st.selectbox('**Select the id column**', dff.columns, dff.index[0])
    longitude = st.selectbox('**Select the column with longitude**', dff.columns, dff.index[0])
    latitude = st.selectbox('**Select the column with latitude**', dff.columns, dff.index[0])
    value = st.selectbox('**Select the column with the attribute for visualisation**', dff.columns, dff.index[0])

    df = dff[[str(longitude), str(latitude), str(value) ]]
    df = df.dropna()
    df.rename(columns={value : "value"}) 
    return df


st.title("Operations with Spatial Data")
st.sidebar.header("Geo Data", divider='rainbow')
st.write("""This demo case illustrates visualisation of data, which includes reference to geographical locations.""")

st.success("Select a file that contains geospatial data, using the file loading tool")
tab = st.file_uploader("Your file")
if tab is not None:  
    try:
        df = readData(tab) 
        st.dataframe(df, use_container_width=True)
        if st.button(":green[Explore]"):
                st.subheader("Explore the Data on Map")           
    except:
        pass
    
st.pydeck_chart(pdk.Deck(
    # map_style=None,     
    initial_view_state=pdk.ViewState(latitude=55.6761, longitude=12.5683, zoom=6, pitch=50),
    layers=[
        pdk.Layer(
           'HexagonLayer',
           data=df,
           get_position='[long, lat]',
           radius=200,
           elevation_scale=4,
           elevation_range=[0, 1000],
           pickable=True,
           extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position='[long, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius='value',
        ),
    ],
))