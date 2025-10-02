import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing


# Assignment 1 Data wrangling and exploration


# Loading the data and check if its loaded correctly
dataFrame = pd.read_csv('../Data/WA_Fn-UseC_-HR-Employee-Attrition.csv', index_col='EmployeeNumber', na_values=['NA'])
print(dataFrame)
print(dataFrame.isnull().sum())

# Reform data into usefull data
# remove columns and turn non-numeric data into numeric data
# Dropping Coulmn Over18, StandardHours and EmployeeCount because they are identical for all employees
# Dropping JobRole because it is represented in the Joblevel column


def preprocessor (dataFrame):
    processedData = dataFrame.copy()
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

    print(processedData.dtypes)

    return processedData

processedData = preprocessor(dataFrame)


np.mean(processedData['Attrition'] == 1)
groupedData = processedData.groupby('Department')["Attrition"].mean()


# Data Mapping, used in DataHandlingGraphs.py
departmentMapping = {
    0: 'Human Resources',
    1: 'Research & Development',
    2: 'Sales'
}


