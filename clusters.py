# ------------------ IMPORTS ------------------
import os
os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans

# ------------------ LOAD DATA ------------------
# Load dataset
# Current working directory
cwd = Path.cwd()
# Path to the CSV file 
csv_file = cwd / "Mall_Customers.csv"
# Load the CSV
df = pd.read_csv(csv_file)

# ------------------ PREVIEW DATA ------------------
# print(df.head())

# ------------------ DROP UNUSED COLUMN ------------------
# CustomerID is just an identifier — remove it
df = df.drop(columns=["CustomerID"])

# ------------------ ENCODING ------------------
le = LabelEncoder()
df["Genre"] = le.fit_transform(df["Genre"])   

#Verify encoded labels
for i, category in enumerate(le.classes_):
    print(f"{category} = {i}")

# ------------------ SELECT FEATURES ------------------
features = ["Genre", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

# ------------------ FEATURE SCALING ------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------ TRAIN K-MEANS------------------
kmeans = KMeans(n_clusters=20, random_state=42) #change the value of n_clusters here
df["Cluster"] = kmeans.fit_predict(X_scaled)

# ------------------ CLUSTER PROFILES ------------------
print("\nCluster Summary:")
print(df.groupby("Cluster").mean())

# ------------------ VISUALIZE CLUSTERS ------------------
plt.figure(figsize=(7,5))
scatter = plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], 
                      c=df["Cluster"], cmap='viridis')
plt.title("Customer Segments by Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1–100)")
plt.colorbar(scatter, label="Cluster")
plt.show()

# ------------------ CLUSTER CENTROIDS ------------------
# centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
# print("\nCluster Centroids (Original Scale):")
# print(centroids)