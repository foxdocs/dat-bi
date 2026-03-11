import readData
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = readData.loadData('../Data/WA_Fn-UseC_-HR-Employee-Attrition.csv', 'csv')
pd.set_option('display.float_format', lambda x: '%.3f' % x)
labelEncoder = preprocessing.LabelEncoder()

jobRoleComparison = data.copy().drop(['EmployeeCount', 
                  'EmployeeNumber', 
                  'Over18', 
                  'StandardHours',
                  'Age',
                  'BusinessTravel',
                  'DailyRate',
                  'Department',
                  'DistanceFromHome',
                  'Education',
                  'EducationField',
                  'EnvironmentSatisfaction',
                  'Gender',
                  'HourlyRate',
                  'JobInvolvement',
                  'JobLevel',
                  'JobSatisfaction',
                  'MaritalStatus',
                  'MonthlyIncome',
                  'MonthlyRate',
                  'NumCompaniesWorked',
                  'OverTime',
                  'PercentSalaryHike',
                  'PerformanceRating',
                  'RelationshipSatisfaction',
                  'StockOptionLevel',
                  'TotalWorkingYears',
                  'TrainingTimesLastYear',
                  'WorkLifeBalance',
                  'YearsAtCompany',
                  'YearsInCurrentRole',
                  'YearsSinceLastPromotion',
                  'YearsWithCurrManager'
                  ], axis=1)

jobRoleComparison['Attrition'] = labelEncoder.fit_transform(jobRoleComparison['Attrition'])
jobRoleComparison.groupby(by='JobRole').mean().plot(kind='bar')
plt.tight_layout()
plt.show()

genderPaymentPerDepartment = data.copy().drop([
                  'Attrition',
                  'EmployeeCount', 
                  'EmployeeNumber', 
                  'Over18', 
                  'StandardHours',
                  'Age',
                  'BusinessTravel',
                  'DailyRate',
                  'DistanceFromHome',
                  'Education',
                  'EducationField',
                  'EnvironmentSatisfaction',
                  'HourlyRate',
                  'JobInvolvement',
                  'JobLevel',
                  'JobRole',
                  'JobSatisfaction',
                  'MaritalStatus',
                  'MonthlyRate',
                  'NumCompaniesWorked',
                  'OverTime',
                  'PercentSalaryHike',
                  'PerformanceRating',
                  'RelationshipSatisfaction',
                  'StockOptionLevel',
                  'TotalWorkingYears',
                  'TrainingTimesLastYear',
                  'WorkLifeBalance',
                  'YearsAtCompany',
                  'YearsInCurrentRole',
                  'YearsSinceLastPromotion',
                  'YearsWithCurrManager'
], axis=1)

genderPaymentPerDepartment['Gender'] = labelEncoder.fit_transform(genderPaymentPerDepartment['Gender'])
#genderPaymentPerDepartment['Department'] = labelEncoder.fit_transform(genderPaymentPerDepartment['Department'])


fig, axes = plt.subplots(3, 1, figsize=(10, 15))

for i, (department, df) in enumerate(genderPaymentPerDepartment.groupby('Department')):
    ax = axes[i]
    df.groupby('Gender')['MonthlyIncome'].mean().plot(kind='bar', ax=ax)
    ax.set_title(f'Average Monthly Pay in Department: {department}')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Average Monthly Pay')
    ax.set_xticklabels(['Female', 'Male'], rotation=0)
plt.tight_layout()
plt.show()