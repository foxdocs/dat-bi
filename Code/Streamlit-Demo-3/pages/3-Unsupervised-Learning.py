import streamlit as st
import pandas as pd
from sklearn import preprocessing
from PIL import Image

df = pd.read_csv('./Data/Employee-Attrition.csv')

processedData = df.copy()
le = preprocessing.LabelEncoder()
processedData['Attrition'] = le.fit_transform(processedData['Attrition'])
processedData['BusinessTravel'] = le.fit_transform(processedData['BusinessTravel'])
processedData['Department'] = le.fit_transform(processedData['Department'])
processedData['EducationField'] = le.fit_transform(processedData['EducationField'])
processedData['Gender'] = le.fit_transform(processedData['Gender'])
processedData['OverTime'] = le.fit_transform(processedData['OverTime'])
processedData['MaritalStatus'] = le.fit_transform(processedData['MaritalStatus'])
processedData = processedData.drop([
        'EmployeeCount',
        'JobRole',
        'Over18',
        'StandardHours'
        ], axis=1)

st.title("Unsupervised Training")
st.sidebar.header("Unsupervised ML", divider='rainbow')

st.write("In this section you can see some of our work with clustering using the K-means method")

st.write("We started by running calculations to find the optimal number of clusters - this was our result:")

st.image(Image.open('./Media/Silhouette Analysis.png'), use_column_width=True)

st.success("From the graph above we learned that the optimal number of clusters would be 3.")

st.write("We then visualised the clusters to get a better overview of how they are seperated.")

st.image(Image.open("./Media/Bounderies of clusters.png"), use_column_width=True)

st.write("And mapped out the silhouette score, to take a look at whether the clusters were good or not.")

st.image(Image.open("./Media/Silhouette Plot of KMeans.png"), use_column_width=True)

st.success("Our silhouette score is 0.49, which is an acceptable result. We imagine that the score would be higher if there wasn't so many entries close to the borders between the clusters.")