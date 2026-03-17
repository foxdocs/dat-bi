import streamlit as st
import joblib

model = joblib.load('./model/nbmodel.pkl')

st.title("Make a prediction")
st.sidebar.header("Make a prediction", divider='rainbow')

st.write("In this section you can use the best trained model we could make to predict if employees has attrition.")

st.success("👇 Fill out the information needed below and press 'train' to get your prediction 👇")

age = st.number_input('Age', 18, 85)
distanceFromHome = st.number_input('Distance from home in miles', 0, 1000)
jobLevel = st.selectbox('Job level', [1, 2, 3, 4, 5])
overTime = st.selectbox('Overtime', ['Yes', 'No'])
timesTrainedLastYear = st.number_input('Times trained in the last year', 0, 12)

button = st.button('Make prediction')

if button: 
    prediction = model.predict([[age, distanceFromHome, jobLevel, 1 if overTime else 0, timesTrainedLastYear]])
    message = 'You have attrition' if prediction==1 else 'You do not have attrition'
    st.write(message)