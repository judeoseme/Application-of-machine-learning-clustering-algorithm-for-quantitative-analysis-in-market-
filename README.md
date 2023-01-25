# Application-of-machine-learning-clustering-algorithm-for-quantitative-analysis-in-market

Introduction:
In today’s world of marketing, a lot of problems are being encountered that calls for immediate action. For instance, how to reduce the high demand of a particular product by bringing in another type, when to launch a particular product, how to influence customers on their choice of consumption, how to make new product available. All these could be handled if the right skills are employed. Clustering algorithm in quantitative model could help to fulfil the required need in an efficient manner. It is an unsupervised machine learning technique that do not require labelled data, and this is one of the advantages over other supervised machine learning like classification and regression. Clustering is a method of dividing data points into several groups such that the data points in the same group are more like other data points in the same group than those in another group. For example, it can be useful for identifying patterns behaviours and developing targeted marketing campaigns. In the context of market and management, clustering algorithm can be used to perform quantitative analysis and gain insights into various business- related phenomena. In addition, it can also be used to identity trends in financial data such as the market patterns. This can be useful to make marketing decision and identify opportunities ft make more profit. Clustering has played a huge role in data mining such as scientific exploration. It has also been used in different fields such as medical diagnostic, web analysis, marketing (Berkhin, 2006). Over the years, there have been a significant change in the field of marketing and management. These changes in marketing and management are because of customer segmentation. In clustering analysis, there is no prediction as the marketing department will always depend on a set of data analytics to improve their productivity and growth. Lefait, G. et al (2010). Generally, in the world of business, the needs of customers are very important. So therefore, it is necessary to find a way to improve sales by evaluating the needs of the customers. 



Datasets:
This dataset is obtained from https://archive.ics.uci.edu/ml/datasets/Wholesale+customers The dataset contains 8 features of which we were able to build a model which separates the customers into different clusters. The eight features are 
1. Channel
2. Region
3. Fresh
4. Milk
5. Grocery
6. Frozen
7. Detergents paper
8. Delicatessen

In this paper, we analysed wholesale customer dataset which consists of annual spending of clients of wholesale distributor on different types of products. We proposed a model for analysis of wholesale customers using K-means and Hierarchical unsupervised learning algorithms. The dataset is taken from UCI data repository, which contains eight features namely Fresh, Milk, Grocery, Frozen, Detergents Paper, Delicatessen, Channel, Region. Based on these features, we build a model which separates the customers into different clusters. Python language was used for implementation of algorithms

Explanation and Preparation of dataset:
Data preparation is a method of combining, gathering, and organizing data so that it can be used for further analysis. It is an important step to undergo when dealing with data. It includes varieties of different task like gathering data from different sources, cleaning, splitting the dataset, etc. Preparation of dataset is important because a well-prepared dataset can improve the performance of machine learning models and other driven data analysis which will help to give a well-informed decision making. In this study, the dataset that was used refers to the yearly spending of money of client’s large quantities distributor. A total of 8 features were used for this analysis namely.
1. Fresh: This refers to the yearly spending of products that are new
2. Milk:    This is the recurring spending on dairy products
3. Grocery: This is the annual spending on consumable products
4. Frozen: This is the yearly laying out (spending) of refrigerated products
5. Detergents paper: Refers to money spent annually on soaps and stationaries
6. Delicatessen: This is the yearly spending of delicatessen products. 
7. Channel
8. Region
Our dataset used in this study were classified into nominal and continuous variables. The continuous variable is 
1. Fresh
2. Milk
3. Grocery
4. Frozen
5. Detergents paper
6. Delicatessen
The following were classified as nominal variable.
1. Channel
2. Region
In this dataset, there are no null or missing values. This could be seen below
 
A brief description of the algorithms used:
In this study, the algorithm that was used is K-means clustering algorithm and DBSCAN algorithm.
1. K-Means Clustering Algorithm: This is an algorithm used in machine learning and data mining to group a dataset into a specified number of clusters. The K-means works by randomly selecting the initialised K centroids and then by allocating each point to a cluster based on which centroids is nearest to it. The algorithm will adjust the centroids and reassign the data points to the appropriate clusters. This process will continue until the cluster’s centroids are no longer changing. The closest centroid to a datapoint can be calculated using the distance metrics known as Euclidean distance. This is given by.
 
The major goal of the K-means algorithm is to split the data into clusters in such a way that the points within individual clusters are similar to each other and are different as possible to pints in other clusters. This algorithm can be used to identify patterns in a dataset and classified data into distinct groups. One major parameter in the K-means algorithm is the value of K, which specifies the number of clusters to be created. 
2.DBSCAN Algorithm: Density-based spatial clustering of applications with noise is an algorithm used in to identify clusters of points in a dataset. The algorithms work by searching for data points that have a high number of neighbouring points within a specified radius and then expanding the clusters to include points that are reachable. This point is known as the core points. A cluster that is not identifiable with any point or that they are too far away from the core points are classified as noise. The DBSCAN has two important parameters. One of the key parameters in the DBSCAN algorithm is the value of epsilon. The epsilon specifies the maximum distance between two points that can be considered neighbors. The other parameter is the minimum number of points required to form a cluster. One of the major advantages of DBSCAN is the ability to handle noise or irregular data. Another important advantage is the ability to identify clusters of different densities.
The application of data-mining techniques to selected datasets that you choose using Python
1.K- means is a clustering technique that is used for finding trends in dataset and to group similar datapoints together. One of the application of data-mining technique in K-means algorithm is the use of elbow method to find the optimal number of clusters. The elbow methods calculate within clusters sum of squares for the data with the various value of K. The aim of K-means clustering is to minimize the value. This will enable us to plot how within clusters sum of square changes as we increase the number of clusters.
 

 
2.Another application is that we can evaluate the performance of the clustering by comparing the clusters using metrics.

Explanation of the experimental procedure, including the setting and optimisation of model hyperparameters during training, and your approach to validation (for supervised learning tasks).
The experimental procedure for K-means involves the following steps. 
1. Pre-processing the data: This includes scaling the data, handling missing values and removing outliers. Basically, pre-processing of data simply refers to cleaning and preparing the data for clustering
2. Choosing the appropriate number of clusters: The first step to consider when choosing the number of clusters, k must be well defined and specified.  One of the common ways to determine the optimal value of K is the elbow method.  
3.  Initializing the centroids: The centres of the clusters, k is called centroids. These centroids must be randomly initialized from data points. 
4. Assigning data points: The data points must be assigned to the nearest centroid based on the distance measure being used. This could be measured using the Euclidean distance. 
5. Updating the centroid: Centroids are being updated to the mean of the points assigned to the corresponding cluster. This is done by recalculating each centroids locations as the centre of all points assigned to its cluster
6. Assigning the data points and updating the centroids must be repeated until the centroids no longer change.
Setting and optimization of model parameters. 
When adjusting the hyperparameters, it is important to consider the number of clusters k. As mentioned above, it must be specified in advance. Also, the distance measure used to determine the similarity between the data points can be adjusted. This could be done using the Euclidean distance. It is also necessary to initialize the centroids, but the method used to perform this could affect the final clusters obtained. Using random initialization will be a good one. 

Approaches to Validation:
One of the several approach to use to evaluate the performance of k-means model is to use the external validation measure like the adjusted R. Another approach is to use an internal validation measure such as the sum of squared distance between the data points and the centroids

Experimental Procedure for DBSAN:
The experimental procedure for DBSCAN involves the following
1. Preprocessing the data:  This includes scaling the data, handling missing values and removing outliers. Basically, pre-processing of data simply refers to cleaning and preparing the data for clustering.
2. Choosing the right values for the hyperparameters: There are two main hyperparameters used in DBCSAN. These are the maximum distance between two points in the same clusters (EPS) and the minimum number of points to form a clusters (MinPts). These two values will always determine the shape and number of the clusters obtained. 
3. Core Points Identification: A point is considered a core point if there is a presence of minimum number of points with a distance of Eps which is the maximum distance between two points. 
4. Assigning Points to Clusters: Points that are core points and within a maximum distance are assigned to the same clusters. All other points are considered noise and are not assigned to clusters. 
5. This process must be repeated until all points are being assigned 
Setting and optimization of model parameters.
One of the approaches that can be used to optimize the hyperparameters of a DBSCAN model is the use of grid search. This is used to test a variety of range of different values of maximum distance between two points in the same clusters and also the minimum number of points. Another approach is the use of elbow method to determine the optimal number of clusters. This information will be useful in the selection of maximum distance between two points and the minimum number of points.
Approaches to Validation
One of the several approach to use to evaluate the performance of DBSCAN model is to use the external validation measure like the adjusted R which helps to compare the clusters obtained. Another approach is to use an internal validation measure such as the density of the clusters which assess the separation of the clusters. 

Visualisation of the results
Seaborn is particularly useful for exploring and visualizing data. It provides us with varieties of scatterplots for each pair of variables and histogram for each variable too. 
 

I applied K-means clustering to the dataset that was used in this study which consisted of 8 features for each sample. We ran K-means with K = 5 clusters and used the elbow method to determine the optimal number of clusters. The resulting clusters are shown above where each colour corresponds to different cluster. The results of the k-means suggest that the data contains 5 clusters as identified by the different colours in visual 3. These clusters correspond to the underlying structure of the data, as evidence by the relatively high score and low within- cluster sum of squares. Further analysis such as examining the feature values within each cluster to known categories could help provide more insights on the nature of these clusters.  Another algorithm that was applied to this study is the DBSCAN. Same dataset was used which consists of 8 features. DBSCAN was ran with Eps = 0.5 and MinPts = 5 . This value was used to evaluate the quality of the resulting clusters. The resulting clusters are shown in visual 4 where each colour corresponds to the different cluster and black colour indicates noise points that were not assigned to any cluster. The results of the DBSCAN analysis suggests that the data contains 6 different clusters apart from the noise since it was not assigned to any clusters as identified by fig. These clusters correspond to the underlying structure of the data as evidenced by relatively high score and low percentage.  It is important to note that the results of the DBSACN analysis are dependent on the values of the Eps and MinPts parameters
Ethical and Legal Considerations:
When discussing the results of the K-means and DBSCAN analysis, it is important to address the ethical and legal consideration. This will help to ensure the ethical use of the data as well as potential risk that could encountered during the analysis. They include. 
1. large volume of data: in this study, the dataset that was used contains only 8 features of   commodities. Which is relatively small to some extent compared to the different products being purchased by people on daily basis. Having a wide range of products will help do a good justice to the clustering activities. 
2. Ethical and legal compliance: In the dataset used in this study, most of the commodities used are consumable products. Therefore, it is necessary to consider the regulations governing the use of data in these industries that produces edible substances. If not well managed, it could influence the sales of the products negatively. 

Conclusion:
In this study, using DBSCAN and k-means means that one can determine patterns and groups similar items in a dataset of customers products. Both algorithms can be use effectively to determine clusters of customers with similar purchasing patterns or to group products that are frequently purchased together. 



References

Berkhin, P. (2006). A Survey of Clustering Data Mining Techniques. In: Kogan, 
J., Nicholas, C., Teboulle, M. (eds) Grouping Multidimensional Data. Springer, Berlin, Heidelberg
Lefait, G., & Kechadi, T. (2010). Customer Segmentation Architecture Based on 
Clustering Techniques. Digital Society. 2010. ICDS 10. Fourth  International Conference on Digital Society. 
