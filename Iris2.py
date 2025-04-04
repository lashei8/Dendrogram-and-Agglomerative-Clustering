import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# Step 1: Load the dataset
df = pd.read_csv("iris.csv")
X = df.iloc[:, :2]  # Use the first and second columns for the x and y axes

# Step 2: Perform Agglomerative Clustering with 3 clusters
clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
cluster_labels = clustering.fit_predict(X)

# Step 3: Remap cluster labels based on the interpretation of the plot
# Green squares (Cluster 1) -> Setosa (0)
# Blue triangles (Cluster 2) -> Versicolor (1)
# Red circles (Cluster 0) -> Virginica (2)
label_mapping = {0: 2, 1: 0, 2: 1}
remapped_labels = [label_mapping[label] for label in cluster_labels]

# Step 4: Create a scatter plot with the remapped labels
colors = ['green', 'blue', 'red']
markers = ['s', '^', 'o']
species = ['Setosa', 'Versicolor', 'Virginica']

plt.figure(figsize=(10, 7))
for i in range(3):
    cluster_points = X[np.array(remapped_labels) == i]
    plt.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], 
                color=colors[i], marker=markers[i], label=species[i])

# Step 5: Annotate each point with its row number
for index, (x, y) in enumerate(zip(X.iloc[:, 0], X.iloc[:, 1]), start=1):
    plt.text(x, y, str(index), fontsize=8, color='black', ha='right')

# Step 6: Set axis labels, title, and legend
plt.xlabel('First Column')
plt.ylabel('Second Column')
plt.title('2D Scatter Plot of Iris Data with Remapped Cluster Labels')
plt.legend(title="Species")
plt.grid(True)
plt.savefig('Iris_2plot.pdf', dpi=300)
plt.show()

