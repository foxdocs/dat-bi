import DataHandling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing

processedData = DataHandling.processedData.copy()

# Create a histogram to show the distribution of the features
DataHandling.processedData.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()
# Create a heatmap to show the correlation between the features
plt.figure(figsize=(20, 15))
sns.heatmap(processedData.corr(), annot=True)
plt.tight_layout()
plt.show()


# Create a histogram to show the distribution of Attrition in the different departments

plt.figure(figsize=(15, 10))

# Map department numbers to department names
processedData['Department'] = processedData['Department'].map(DataHandling.departmentMapping)

# Count of total employees in each department
totalCounts = sns.countplot(data=processedData, x='Department', palette="Set3", label='Total Employees')

# Count of employees with Attrition in each department
attritionCount = sns.countplot(data=processedData[processedData['Attrition'] == 1], x='Department', palette="Set2", label='Employees with Attrition')

# Extract counts for each department
total_counts_by_department = processedData.groupby('Department').size()
attrition_counts_by_department = processedData[processedData['Attrition'] == 1].groupby('Department').size()

# Calculate and annotate with counts and percentages
for c, a, department in zip(totalCounts.patches, attritionCount.patches, total_counts_by_department.index):
    total_count = total_counts_by_department[department]
    attrition_count = attrition_counts_by_department.get(department, 0)
    percentage = (attrition_count / total_count) * 100 if total_count != 0 else 0

    # Annotate with actual counts
    totalCounts.annotate(f'{total_count}', (c.get_x() + c.get_width() / 2., c.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # Annotate with counts and percentages
    attritionCount.annotate(f'{attrition_count}\n({percentage:.2f}%)', (a.get_x() + a.get_width() / 2., attrition_count), ha='center', va='center', xytext=(0,10), textcoords='offset points')

plt.title('Attrition in the different departments')
plt.xlabel('Department')
plt.ylabel('Count')
plt.legend(title='Attrition in Departments', loc='upper right')
plt.show()



# create a histogram to show how long people have been employed by the company
plt.figure(figsize=(15, 10))
sns.histplot(data=processedData, x='YearsAtCompany', kde=True)
plt.title('Years employed by the Company')
plt.xlabel('Years at the Company')
plt.ylabel('Count')
plt.show()

# create a histogram to show the general age distribution of the employees
plt.figure(figsize=(15, 10))
sns.histplot(data=processedData, x='Age', kde=True)
plt.title('Age Distribution of Employees')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()