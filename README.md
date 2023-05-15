# Madelon Dataset: PCA and K-means Clustering

This project explores the Madelon dataset, a synthetic high-dimensional dataset with 500 features and 2,600 data points. The goal is to perform dimensionality reduction using PCA and apply k-means clustering to gain insights into the dataset's structure.

## Dataset

The Madelon dataset is a synthetic dataset generated for a machine learning competition. It has a highly non-linear structure and poses challenges for clustering algorithms due to its high dimensionality and potential noise. The dataset contains 500 features and 2,600 data points.

## Preprocessing

To prepare the data for clustering, we applied the following preprocessing techniques:

- Standardization: The dataset was standardized using the StandardScaler from scikit-learn. This step scales the features and brings them to a similar range, which is particularly important for high-dimensional datasets.

## K-means Clustering

We applied k-means clustering to the Madelon dataset with different values of k: 4, 8, 16, 32, and 64. The main objective was to analyze how the number of clusters affects the performance of the clustering algorithm.

- Performance Analysis: We measured the performance of the k-means algorithm using the sum of squared distance (SSE) between data points and their assigned centroids. We generated a plot of k values (x-axis) versus SSE (y-axis) to identify the optimal k value.

- Optimal k: Based on the SSE plot, we determined the optimal k value. This value indicates the number of clusters that best captures the underlying structure of the dataset.

- Initialization of Centroids: For k = 8, we initialized the set of cluster centroids randomly before running k-means. We also reran k-means with different initial centroids to observe the impact on the final clusters.

## Evaluation Metrics for Clustering

To assess the quality of the clusters produced by k-means or any other clustering algorithm, several evaluation metrics can be used, such as:

- Silhouette Score: Measures the cohesion and separation of clusters.
- Davies-Bouldin Index: Evaluates the compactness and separation between clusters.
- Calinski-Harabasz Index: Measures the ratio of between-cluster dispersion to within-cluster dispersion.

## Principal Component Analysis (PCA)

To address the high dimensionality of the Madelon dataset, we performed PCA, which is a popular technique for dimensionality reduction.

- Cumulative Variance Plot: We created a plot of the number of components included (x-axis) versus the amount of variance captured (y-axis). Our goal was to retain at least 75% of the variance to ensure meaningful representation.

- Selected Principal Components: We selected a certain number of principal components based on the cumulative variance plot and performed PCA using those components.

- Transformed Data Plot: We plotted the transformed data on a graph using the first two principal components as the axes (PC1 and PC2).

## Scatter Plot Analysis

We examined the scatter plot of PC1 (x-axis) versus PC2 (y-axis) to observe any visible clusters in the transformed data. The presence or absence of clusters in this plot can provide insights into the dataset's structure.

## K-means Clustering on Transformed Data

We performed k-means clustering on the transformed data obtained from PCA. This approach has several advantages, especially for high-dimensional datasets:

- Reduced Dimensionality: PCA reduces the dimensionality of the data, making it more manageable for clustering algorithms.
- Enhanced Separation: PCA can improve the separation between clusters by identifying the most informative features.

- Elbow Plot: We generated an elbow plot using the same k values as before (4, 8, 16# Madelon Dataset: PCA and K-means Clustering

This project explores the Madelon dataset, a synthetic high-dimensional dataset with 500 features and 2,600 data points. The goal is to perform dimensionality reduction using PCA and apply k-means clustering to gain insights into the dataset's structure.

## Dataset

The Madelon dataset is a synthetic dataset generated for a machine learning competition. It has a highly non-linear structure and poses challenges for clustering algorithms due to its high dimensionality and potential noise. The dataset contains 500 features and 2,600 data points.

## Preprocessing

To prepare the data for clustering, we applied the following preprocessing techniques:

- Standardization: The dataset was standardized using the StandardScaler from scikit-learn. This step scales the features and brings them to a similar range, which is particularly important for high-dimensional datasets.

## K-means Clustering

We applied k-means clustering to the Madelon dataset with different values of k: 4, 8, 16, 32, and 64. The main objective was to analyze how the number of clusters affects the performance of the clustering algorithm.

- Performance Analysis: We measured the performance of the k-means algorithm using the sum of squared distance (SSE) between data points and their assigned centroids. We generated a plot of k values (x-axis) versus SSE (y-axis) to identify the optimal k value.

- Optimal k: Based on the SSE plot, we determined the optimal k value. This value indicates the number of clusters that best captures the underlying structure of the dataset.

- Initialization of Centroids: For k = 8, we initialized the set of cluster centroids randomly before running k-means. We also reran k-means with different initial centroids to observe the impact on the final clusters.

## Evaluation Metrics for Clustering

To assess the quality of the clusters produced by k-means or any other clustering algorithm, several evaluation metrics can be used, such as:

- Silhouette Score: Measures the cohesion and separation of clusters.
- Davies-Bouldin Index: Evaluates the compactness and separation between clusters.
- Calinski-Harabasz Index: Measures the ratio of between-cluster dispersion to within-cluster dispersion.

## Principal Component Analysis (PCA)

To address the high dimensionality of the Madelon dataset, we performed PCA, which is a popular technique for dimensionality reduction.

- Cumulative Variance Plot: We created a plot of the number of components included (x-axis) versus the amount of variance captured (y-axis). Our goal was to retain at least 75% of the variance to ensure meaningful representation.

- Selected Principal Components: We selected a certain number of principal components based on the cumulative variance plot and performed PCA using those components.

- Transformed Data Plot: We plotted the transformed data on a graph using the first two principal components as the axes (PC1 and PC2).

## Scatter Plot Analysis

We examined the scatter plot of PC1 (x-axis) versus PC2 (y-axis) to observe any visible clusters in the transformed data. The presence or absence of clusters in this plot can provide insights into the dataset's structure.

## K-means Clustering on Transformed Data

We performed k-means clustering on the transformed data obtained from PCA. This approach has several advantages, especially for high-dimensional datasets:

- Reduced Dimensionality: PCA reduces the dimensionality of the data, making it more manageable for clustering algorithms.
- Enhanced Separation: PCA can improve the separation between clusters by identifying the most informative features.

- Elbow Plot: We generated an elbow plot using the same k values as before  (4, 8, 16, 32, 64).  
- Compare the SSE values plotted in this exercise to the previous plot before performing PCA.
