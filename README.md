# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program and load the dataset. Import the Mall Customers dataset and read the data required for the analysis.

2. Select important features from the dataset. Choose relevant attributes such as Annual Income and Spending Score which help in grouping customers.

3. Choose the number of clusters (K). Decide the number of customer groups using a suitable method such as the elbow method.

4. Initialize and apply the K-Means algorithm. The algorithm randomly selects cluster centers and calculates the distance between data points and centers.

5. Assign customers to the nearest cluster. Each customer is grouped based on similarity in purchasing behavior and the cluster centers are updated repeatedly.

6. Display the final clusters and analyze the result. Visualize the groups of customers and interpret their purchasing habits for better marketing strategies. 

## Program:
```
/*
Program to implement customer segmentation using K-Means clustering on the Mall Customers dataset.
Developed by: S Hariharan
RegisterNumber: 212225040109
*/
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

data = pd.read_csv('CustomerData.csv')

print(data.head())
print(data.columns)

features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)  # Explicit n_init to suppress warning
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)
    
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)  # Explicit n_init
kmeans.fit(X_scaled)

data['Cluster'] = kmeans.labels_

sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')

print("\nName: S HARIHARAN")
print("Reg No.: 212225040109\n")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data,x='Annual Income (k$)',y='Spending Score (1-100)',hue='Cluster', palette='viridis',s=100,alpha=0.7)
plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
```

## Output:

![alt text](<Screenshot 2026-03-11 175705.png>)

![alt text](<Screenshot 2026-03-11 175803.png>)

![alt text](<Screenshot 2026-03-11 175816.png>)

## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
