# MP4-MACHINE-LEARNING-FOR-ANALYSIS-AND-PREDICTION-OF-ATTRITION
Mini Projekt 4 - Business Intelligence
made by Carsten Juhl, Danyal Kitir, Pelle Hald Vedsmand and Nicolai Rosendahl

### Which machine learning methods did you choose to apply in the application?

We tried training models using the decision tree, naive bayes and clustering method K-means. We chose to implement the naive bayes model, because it yielded the overall best results. 

### How accurate is your solution of prediction?

Our model is 88% accurate. Even though that is decent result overall, we have to take into account that the recall of the 'yes' prediction is 0.35, meaning that the model is prone to false negatives. This could be due to class imbalance of the data set, or possibly lack of traning data. It was however the best result we got. 

### Which are the most decisive factors for quitting a job?

We found that the features that matters most in regards to attrition is 'Age', 'DistanceFromHome, 'JobLevel', 'OverTime' and 'TrainingTimesLastYear'.

### Which work positions and departments are in higher risk of losing employees?

The department at highest risk of losing employees due to attrition is HR, with an attrition percentage of 20.63.

<image src='/Data/Exploration/Attrition shown in different department.png'>

The position with the highest risk of losing employees due to attrition is sales representative with almost 40% having attrition.

<image src='/Data/Exploration/AttritionPerJobRole.png'>

### Are employees of different gender paid equally in all departments?

On average women are paid a little more than men in Human Resources and Research and Development, while the average pay is fairly equal in the sales department. 

<image src='/Data/Exploration/MonthlyPayPerGenderPerDerpartment.png'>

### Do the family status and the distance from work influence the work-life balance?

According to our heatmap there isn't a notable correlation between family status, distance from work and work-life balance. 

### Does education make people happy (satisfied from the work)?

According to our heatmap there isn't a notable correlation between education or education field and jobsatisfaction.

<image src='/Data/Exploration/Heatmap of correlation between all features.png'>

### Which were the challenges in the project development?

Figuring out which features had the biggest influence on attrition. We found that it is difficult to predict which features matter most because the human brain can't see patterns across multiple planes like a computer can. We therefore had to rely a lot on a combination of human intuition and using the computer to calculate as many options as possible. 