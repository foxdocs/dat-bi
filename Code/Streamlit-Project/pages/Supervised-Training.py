import streamlit as st
import pandas as pd
from PIL import Image
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import classification_report

df = pd.read_csv('./Data/WA_Fn-UseC_-HR-Employee-Attrition.csv')

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
        'StandardHours',
        'Attrition'
        ], axis=1)

st.title("Supervised Training")
st.sidebar.header("Supervised Training", divider='rainbow')

st.write("In this section you can train a model using supervised training models.")

st.success("👇 Select a method that you want to train your model with and which features you want included in your model 👇")

method = st.selectbox("**Choose a method and features to train with:**", ['Decision Tree', 'Naive Bayes'])

allCheckboxes = []

for feature in processedData.columns:
    checkbox = st.checkbox(feature, False)
    allCheckboxes.append(checkbox)

trainButton = st.button('**Train model**')

if trainButton: 
    trainingData = processedData.copy()
    for index, checkbox in enumerate(allCheckboxes):
        if not checkbox:
            trainingData = trainingData.drop([processedData.columns[index]], axis=1)
    chosenMethod = None    
    if method == 'Decision Tree':
        chosenMethod = DecisionTreeClassifier()
    elif method == 'Naive Bayes':
        chosenMethod = GaussianNB()
        
    X_train, X_test, y_train, y_test = model_selection.train_test_split(trainingData, df['Attrition'], test_size=0.2, random_state=5)

    chosenMethod.fit(X_train, y_train)

    predictions = chosenMethod.predict(X_test)

    st.write(classification_report(y_test, predictions))