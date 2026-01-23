import streamlit as st
import joblib

model = joblib.load('./model/nbmodel.pkl')

st.title("Make a Prediction")
st.sidebar.header("Make a Prediction", divider='rainbow')

st.write("We have trained a  model to predict your job attitude.")

st.success("👇 Fill out the information needed below and press '[predict]' to get your prediction 👇")

# age = st.slider("Your age?", 18, 60)
age = st.number_input('Age', 18, 85)
distanceFromHome = st.slider('How far is home from work in km', 0, 25)
overTime = st.radio("Do you work overtime", ["Yes", "No"],index=None)
# overTime = st.selectbox('Overtime', ['Yes', 'No'])
jobLevel = st.selectbox('Your job level?', [1, 2, 3, 4, 5])
timesTrainedLastYear = st.number_input('Times trained in the last year', 0, 12)

button = st.button('Predict')

if button: 
    prediction = model.predict([[age, distanceFromHome, jobLevel, 1 if overTime else 0, timesTrainedLastYear]])
    message = 'You are going to quit your job' if prediction==1 else 'You are not going to quit your job'
    st.write(message)