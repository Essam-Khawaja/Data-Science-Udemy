# Cluster Analysis

Official Definition:
A multivariate statistical technique that groups observations on the basis some of their features or variables they are described by.

But the goal of clustering, at the end of the day, is to maximize the similarity of observations within a cluster and maximize the dissimilarity between clusters.

This is often the first step to any case study; often used with other methods.

For classification, such as in Logistic Regression, we often already have labelled data. This is a supervised learning method. Clustering on the other hand, is an unsupervised learning method. We have no clue what the right output is, so we simply try and find the patterns in our data to gain insights out of.

## Math
### Euclidean Distance
We need to know how to calculate the euclidean distance between two points:

$$
d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}
$$

This applies to all dimensions, we just have to find the square root of the sum of the squared difference between the dimensions. We also know the distance from one point A to another point B, is the same as the distance from B to A: 

$$
d(A, B) = d(B, A) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2 + ...}
$$

### Centroids
Mean position of a group of points (the center of mass).

## K-Means Clustering
1. Choose the number of clusters (k)
2. Specify the cluster seeds (A seed is a starting centroid). Either it is assigned randomly, or a data scientist assigns it based on previous knowledge.
3. Assign each point on the graph to one seed. This is done by calculating the distance from each point to each seed, then choosing the minimum distance (the closest seed) as the assignment.
4. Adjust the centroids. We will move the seeds closer to the clusters so that they are actual centroids to the clusters.

Note that the last two steps are repeated; until we can no longer re-classify the points. This is because every time we adjust the seeds, some points may be assigned to different clusters based on the new position. We need to repeat the clustering until no point is changing clusters anymore. 

### How to select the number of clusters: The Elbow Method
The distance between points in a cluster is called 'within-cluster sum of squares', which is WCSS for short. If we minimize WCSS, then we have the perfect solution. We want WCSS to get as low as possible while still retaining meaning in the overall data. The elbow method comes around when we plot WCSS against number of clusters, and notice that at two points in the graph, there is a shift in performance (called the elbow).

### Pros and Cons of KMeans
#### Pros:
1. Simple to understand
2. Fast to cluster
3. Widely available
4. Easy to implement
5. Always yields a result (Could also be a con because result could be deceiving)

#### Cons:
1. We need to pick K (Remedy: Elbow Method)
2. Sensitive to initialization: If original seeds are not good, then entire solution may end up being meaningless. (Remedy: k-means++; another algorithm is run before kmeans to determine the best initial seeds)
3. Sensitive to outliers: If outlier is present, it almost always ends up in a separate cluster to the others. (Remedy: Remove the outliers before running kmeans)
4. Produces spherical solutions instead of elliptic; not too bad but could lead to bad judgement in certain cases
5. Standardization: Not standardizing the data will lead the clusters to be determined partially by the range of the data, for if the range is higher, more weight may be placed onto higher values. As such, we remove that by simply standardizing before the clustering. Now, sometimes we also have to not standardize, such as in cases where we know a variable is more important than the other (Standardization is meant to put all the variables on equal footing). 