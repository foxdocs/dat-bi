import readData
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import joblib

data = readData.loadData('../Data/WA_Fn-UseC_-HR-Employee-Attrition.csv', 'csv')
pd.set_option('display.float_format', lambda x: '%.3f' % x)

print(data.shape)
print(data.describe())
data = data.drop(['EmployeeCount', 
                  'EmployeeNumber', 
                  'Over18', 
                  'StandardHours',
                  #'Age',
                  #'BusinessTravel',
                  'DailyRate',
                  #'Department',
                  #'DistanceFromHome',
                  'Education',
                  'EducationField',
                  'EnvironmentSatisfaction',
                  #'Gender',
                  'HourlyRate',
                  'JobInvolvement',
                  #'JobLevel',
                  'JobRole',
                  #'JobSatisfaction',
                  'MaritalStatus',
                  #'MonthlyIncome',
                  'MonthlyRate',
                  'NumCompaniesWorked',
                  #'OverTime',
                  'PercentSalaryHike',
                  #'PerformanceRating',
                  'RelationshipSatisfaction',
                  #'StockOptionLevel',
                  'TotalWorkingYears',
                  #'TrainingTimesLastYear',
                  #'WorkLifeBalance',
                  #'YearsAtCompany',
                  'YearsInCurrentRole',
                  #'YearsSinceLastPromotion',
                  'YearsWithCurrManager'
                  ], axis=1)
print(data.dtypes)

print(data.isnull().sum())

# No data is null.

labelEncoder = preprocessing.LabelEncoder()
numericDataFrame = data.copy()

numericDataFrame['BusinessTravel'] = labelEncoder.fit_transform(data['BusinessTravel'])
numericDataFrame['Department'] = labelEncoder.fit_transform(data['Department'])
#numericDataFrame['EducationField'] = labelEncoder.fit_transform(data['EducationField'])
numericDataFrame['Gender'] = labelEncoder.fit_transform(data['Gender'])
#numericDataFrame['JobRole'] = labelEncoder.fit_transform(data['JobRole'])
#numericDataFrame['MaritalStatus'] = labelEncoder.fit_transform(data['MaritalStatus'])
numericDataFrame['OverTime'] = labelEncoder.fit_transform(data['OverTime'])
numericDataFrame['Attrition'] = labelEncoder.fit_transform(data['Attrition'])
'''
correlation_matrix = numericDataFrame.corr()
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, annot=True)
plt.tight_layout()
plt.show()
'''
'''
# Trying PCA attempt to reduce the number of features for performance reasons.
sc = StandardScaler()
stanardisedData = sc.fit_transform(numericDataFrame.drop(['Attrition'], axis=1))

pca = PCA(n_components=14)
components = pca.fit_transform(stanardisedData)

plt.figure(figsize=(10, 8))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance Ratio')
plt.show()

# Finding the optimal number of components (We use 95% as the threshold for the explained variance ratio)
for i, explainedVariance in enumerate(np.cumsum(pca.explained_variance_ratio_)):
    if explainedVariance > 0.95:
        print(f'The optimal number of components is {i + 1}')
        break
'''
X_train, X_test, y_train, y_test = model_selection.train_test_split(numericDataFrame[['Age', 'DistanceFromHome', 'JobLevel', 'OverTime', 'TrainingTimesLastYear']], numericDataFrame['Attrition'], test_size=0.2, random_state=5)

decisionTree = DecisionTreeClassifier(max_depth=5, random_state=42)
naiveBayes = GaussianNB()

decisionTree.fit(X_train, y_train)
naiveBayes.fit(X_train, y_train)

dTreePredictions = decisionTree.predict(X_test)
naiveBayesPredictions = naiveBayes.predict(X_test)

dTreeAccuracy = accuracy_score(y_test, dTreePredictions)
naiveBayesAccuracy = accuracy_score(y_test, naiveBayesPredictions)

print(dTreeAccuracy)
print(classification_report(y_test, dTreePredictions))
print(naiveBayesAccuracy)
print(classification_report(y_test, naiveBayesPredictions))

joblib.dump(naiveBayes, '../model/nbmodel.pkl')

def findBestFeatureCombinationForBestPossibleModelTraining(X, y, models):
    best_model = None
    best_accuracy = 0
    best_combination_of_features = None

    features = X.columns.tolist()

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=5)

    for r in range(1, len(features) + 1):
        print(len(features) +1)
        for feature_combination in itertools.combinations(features, r):
            feature_combination = list(feature_combination)

            for model_name, model_instance in models:
                model_instance.fit(X_train[feature_combination], y_train)

                predictions = model_instance.predict(X_test[feature_combination])

                accuracy = accuracy_score(y_test, predictions)

                if accuracy > best_accuracy:
                    best_model = model_name
                    best_accuracy = accuracy
                    best_combination_of_features = feature_combination
        print(r)

    return best_model, best_accuracy, best_combination_of_features

models = [('decisionTree', DecisionTreeClassifier()), ('naiveBayes', GaussianNB())]
bestModel, bestAccuracy, bestFeatureCombination = findBestFeatureCombinationForBestPossibleModelTraining(numericDataFrame.drop(['Attrition'], axis=1), numericDataFrame['Attrition'], models)

print(f'Best model: {bestModel}')
print(f'Best accuracy: {bestAccuracy}')
print(f'Best combination of features: {bestFeatureCombination}')