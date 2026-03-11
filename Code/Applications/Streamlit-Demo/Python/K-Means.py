import DataHandling

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from yellowbrick.cluster import SilhouetteVisualizer

X = DataHandling.processedData.copy()

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

silScore = []

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    clusterLabels = kmeans.labels_
    silhouette_avg = silhouette_score(X, clusterLabels)
    silScore.append(silhouette_avg)


plt.plot(range(2, 11), silScore)
plt.title('Silhouette Analysis')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()


optimalK = 3

kmeans = KMeans(n_clusters=optimalK, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X)

clusterLabels = kmeans.labels_

silhouette_avg = silhouette_score(X, clusterLabels)

# The Silhouette score is a measure of how similar an object is to its own cluster compared to other clusters
# The score ranges from -1 to 1, where the higher the value, the better the clustering
# this cluster has a score of 0.49 which is an acceptable score
print('Silhouette score = ', silhouette_avg)

y = kmeans.predict(X)

# Plotting the clusters 

for i in range(optimalK):
    cluster = X_pca[y == i]

    print ("Cluster ", i, ":", cluster.shape)

    plt.scatter(cluster[:, 0], cluster[:,1], label=f'Cluster {i}')

plt.legend()
plt.grid(True)
plt.show()

# Print the clusters with boundaries

x_min, x_max = X_pca[:, 0].min(), X_pca[:, 0].max()
y_min, y_max = X_pca[:, 1].min(), X_pca[:, 1].max()

n_samples = 1000  # Adjust the number of samples as needed

# Create a meshgrid
xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_samples),
                     np.linspace(y_min, y_max, n_samples))

# Transform meshgrid points back to original feature space
meshgrid_points = pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()])

# Predict cluster labels for meshgrid points
labels = kmeans.predict(meshgrid_points)
labels = labels.reshape(xx.shape)

plt.figure()
plt.clf()
plt.title('Boundaries of the clusters')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.imshow(labels, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired, aspect='auto', origin='lower')

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, s=30, cmap=plt.cm.Paired, edgecolors='k')

# Transform cluster centers back to original feature space using the original data
centers_original = pca.inverse_transform(kmeans.cluster_centers_[:, :2])

plt.scatter(centers_original[:, 0], centers_original[:, 1], c='red', s=200, alpha=0.5, marker='*', facecolors='black')

for i, center in enumerate(centers_original):
    plt.annotate(str(i), tuple(center[:2]), size=20, zorder=1, color='white', horizontalalignment='center', verticalalignment='center')

plt.show()



model = KMeans(n_clusters=optimalK, n_init=10)
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')

# Fit the visualizer
visualizer.fit(X)

# Show the visualizer
visualizer.show()


attrition_labels = DataHandling.processedData['Attrition']

for i in range(optimalK):
    cluster = X_pca[y == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i}', c=attrition_labels[y == i], cmap='viridis')

plt.legend()
plt.text(0.95, 0.05, 'People with attrition = yellow', verticalalignment='bottom', horizontalalignment='right',
         transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7))
plt.grid(True)
plt.show()

y = DataHandling.processedData['Attrition']
X_features = DataHandling.processedData.drop('Attrition', axis=1)

# Train a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=0)
rf_model.fit(X_features, y)

# Display feature importance
feature_importance = pd.Series(rf_model.feature_importances_, index=X_features.columns)
feature_importance.sort_values(ascending=False, inplace=True)
print("Feature Importance:")
print(feature_importance.head())

DataHandling.processedData['Cluster'] = clusterLabels

# Group by cluster and calculate mean values
cluster_means = DataHandling.processedData.groupby('Cluster').mean()

# Display mean values for each cluster
pd.set_option('display.max_columns', None)
print("Mean Values for Each Cluster:")
print(cluster_means)