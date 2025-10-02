import streamlit as st
import pandas as pd
from PIL import Image
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

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
        'StandardHours',
        'Attrition'
        ], axis=1)

st.title("Supervised ML")
st.sidebar.header("Supervised ML", divider='rainbow')

st.write("In this section you can train a model using supervised machine learning models.")

st.success("👇 Select a method that you want to train your model with and which features you want included in your model 👇")

method = st.selectbox("**Choose a method and features to train with:**", ['Decision Tree', 'Naive Bayes'])

allCheckboxes = []
chosenMethod = None 

y = df['Attrition']
target_names = y.unique()

for feature in processedData.columns:
    checkbox = st.checkbox(feature, False)
    allCheckboxes.append(checkbox)

trainButton = st.button('**Training a Model**')

if trainButton: 
    trainingData = processedData.copy()
    for index, checkbox in enumerate(allCheckboxes):
        if not checkbox:
            X = trainingData.drop([processedData.columns[index]], axis=1)
       
    if method == 'Decision Tree':
        chosenMethod = DecisionTreeClassifier()
    elif method == 'Naive Bayes':
        chosenMethod = GaussianNB()
        
    # Apply oversampling using RandomOverSampler
    oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X, y)
        
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=5)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=5)

    chosenMethod.fit(X_train, y_train)

    predicted = chosenMethod.predict(X_test)
    
    st.write("Model Validation")

    # st.write(classification_report(y_test, predicted))
    
    st.dataframe(pd.DataFrame (classification_report(y_test, predicted, target_names=target_names, output_dict=True)).transpose()
)
    
    
    