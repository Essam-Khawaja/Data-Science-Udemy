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


