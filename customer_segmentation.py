# ==========================================
# CUSTOMER SEGMENTATION ANALYSIS
# ==========================================

# STEP 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================================
# STEP 2: Load Dataset
# ==========================================

df = pd.read_csv("customer_segmentation.csv")

print("\nDataset Loaded Successfully\n")

# ==========================================
# STEP 3: Display First 5 Rows
# ==========================================

print(df.head())

# ==========================================
# STEP 4: Display Column Names
# ==========================================

print("\nColumn Names:")
print(df.columns)

# Remove extra spaces from column names

df.columns = df.columns.str.strip()

# Rename columns if using Kaggle Mall Dataset

df.rename(columns={
    "Customer ID": "Customer ID",
    "CustomerID": "Customer ID",
    "Annual Income (k$)": "Annual_Income",
    "Annual Income": "Annual_Income",
    "Spending Score (1-100)": "Spending_Score",
    "Spending Score": "Spending_Score"
}, inplace=True)

print("\nUpdated Column Names:")
print(df.columns)

# ==========================================
# STEP 5: Dataset Information
# ==========================================

print("\nDataset Information")
print(df.info())

# ==========================================
# STEP 6: Statistical Summary
# ==========================================

print("\nStatistical Summary")
print(df.describe())

# ==========================================
# STEP 7: Check Missing Values
# ==========================================

print("\nMissing Values")
print(df.isnull().sum())

# ==========================================
# STEP 8: Remove Duplicate Rows
# ==========================================

df.drop_duplicates(inplace=True)

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

# ==========================================
# STEP 9: Exploratory Data Analysis
# ==========================================

# Annual Income Distribution

plt.figure(figsize=(8,5))
sns.histplot(df["Annual_Income"], bins=20, kde=True)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income")
plt.ylabel("Count")
plt.show()

# Spending Score Distribution

plt.figure(figsize=(8,5))
sns.histplot(df["Spending_Score"], bins=20, kde=True)
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Count")
plt.show()

# Age Distribution

plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Gender Distribution

plt.figure(figsize=(6,5))
sns.countplot(x="Gender", data=df)
plt.title("Gender Distribution")
plt.show()

# ==========================================
# STEP 10: Correlation Heatmap
# ==========================================

plt.figure(figsize=(6,5))

numeric_df = df.select_dtypes(include=['number'])

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# ==========================================
# STEP 11: Select Features
# ==========================================

X = df[["Annual_Income", "Spending_Score"]]

# ==========================================
# STEP 12: Feature Scaling
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# STEP 13: Elbow Method
# ==========================================

wcss = []

for i in range(1,11):

    model = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    model.fit(X_scaled)

    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    wcss,
    marker='o'
)

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.show()

# ==========================================
# STEP 14: Apply KMeans
# ==========================================

kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

# ==========================================
# STEP 15: Display Clustered Dataset
# ==========================================

print("\nClustered Dataset")
print(df.head())

# ==========================================
# STEP 16: Cluster Visualization
# ==========================================

plt.figure(figsize=(10,7))

sns.scatterplot(
    x=df["Annual_Income"],
    y=df["Spending_Score"],
    hue=df["Cluster"],
    palette="Set1",
    s=120
)

centers = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centers[:,0],
    centers[:,1],
    c='black',
    s=300,
    marker='X',
    label='Centroids'
)

plt.title("Customer Segmentation using K-Means")

plt.xlabel("Annual Income")

plt.ylabel("Spending Score")

plt.legend()

plt.show()

# ==========================================
# STEP 17: Cluster Centers
# ==========================================

cluster_centers = pd.DataFrame(
    centers,
    columns=["Annual_Income","Spending_Score"]
)

print("\nCluster Centers")
print(cluster_centers)

# ==========================================
# STEP 18: Cluster Summary
# ==========================================

summary = df.groupby("Cluster")[["Age","Annual_Income","Spending_Score"]].mean()

print("\nCluster Summary")
print(summary)

# ==========================================
# STEP 19: Number of Customers in Each Cluster
# ==========================================

print("\nCustomers in Each Cluster")

print(df["Cluster"].value_counts())

# ==========================================
# STEP 20: Save Output
# ==========================================

df.to_csv(
    "customer_segments_output.csv",
    index=False
)

print("\nOutput File Saved Successfully!")

# ==========================================
# STEP 21: Business Insights
# ==========================================

print("\nBusiness Interpretation")

print("""
Cluster 0:
Low Income - Low Spending
Target with discounts and coupons.

Cluster 1:
High Income - High Spending
Premium customers.
Provide VIP memberships and exclusive offers.

Cluster 2:
Low Income - High Spending
Impulse buyers.
Recommend attractive product bundles.

Cluster 3:
High Income - Low Spending
Potential customers.
Improve engagement through personalized marketing.

Cluster 4:
Medium Income - Medium Spending
Regular customers.
Maintain loyalty through reward programs.
""")
