import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Step 1: Load the dataset
df = pd.read_csv("iris.csv")
X = df.iloc[:, :4]  # First four columns (features)

# Step 2: Perform Agglomerative Clustering using Ward's method
Z_linkage = linkage(X, method='ward', metric='euclidean')

# Step 3: Plot the dendrogram
plt.figure(figsize=(10, 7))
plt.title("Dendrogram for Iris Dataset")
dendrogram(Z_linkage)
plt.xlabel("x_Index")
plt.ylabel("Distance (ward's method)")
plt.savefig('IrisDendrogram_plot.pdf', dpi=300)
plt.show()

# Step 4: Assign cluster IDs (0, 1, or 2) based on the dendrogram (using 3 clusters)
cluster_labels = fcluster(Z_linkage, t=3, criterion='maxclust') - 1

# Step 5: Create a DataFrame with Row Numbers and Cluster IDs
correspondence_table = pd.DataFrame({
    "Row Number": range(1, len(cluster_labels) + 1),
    "Cluster ID": cluster_labels
})

# Step 6: Split the table into three equal parts and format the output
part_size = len(cluster_labels) // 3
split1 = correspondence_table["Cluster ID"][:part_size]
split2 = correspondence_table["Cluster ID"][part_size:2*part_size]
split3 = correspondence_table["Cluster ID"][2*part_size:]

# Create the final formatted table
formatted_table = pd.DataFrame({
    "1": split1.values,
    "2": split2.values,
    "3": split3.values
})

# Add the Row Numbers as the index (1–50 for each split)
formatted_table.index = range(1, part_size + 1)

# Display the formatted table
print(formatted_table.head(10))

# Optionally save the table to a CSV file
formatted_table.to_csv("formatted_cluster_table.csv", index_label="Row Number")

