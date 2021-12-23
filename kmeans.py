import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.model_selection import train_test_split


def k_init(X, k):
    """ k-means++: initialization algorithm

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters


    """
    #I am going to return C, the array of means for the clusters
    #I am also going to return I, the labels of the cluster the corresponding row in X belongs to

    C = []

    I=[]
    # randomly assign to I from 0 to k for each row in I

    for row in X:
        I.append(np.random.randint(k ))

    for cluster in range(0, k):
        # select those rows from X that have I = k
        temps1 = []
        temps2 = []
        for row in range(0, len(I)):
            if I[row] == cluster:
                temps1.append(X[row][0])
                temps2.append(X[row][1])
        # find the mean of those rows and assign it to C[k]
        # mean 
        if pd.isna(np.mean(temps1)) or pd.isna(np.mean(temps2)):
            C.append([0, 0])
        else:
            C.append([np.mean(temps1), np.mean(temps2)])

    return I, C




def assign_data2clusters(X, C, I, k):
    """ Assignments of data to the clusters
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    """
    #I am going to return list I which is the same length as the numver of observations in X
    # and contains the label of the cluster that corresponding row of X belongs to  
    distances = []
    I = []
    for row in range(len(X)):
        for mean in C:
            #find distance between X and mean
            dist = (mean[0] - X[row][0])**2 + (mean[1] - X[row][1])**2
            #append to list of distances
            distances.append(dist)
        #find the minimum of that list of distances
        smallestDist = distances.index(min(distances))
        #the index of the minimum will be assigned to i at the row of xObservation
        I.append(smallestDist)
        distances = []
    C = []
    for cluster in range(k):
        # select those rows from X that have I = k
        temps1 = []
        temps2 = []
        for row in range(0, len(I)):
            if I[row] == cluster:
                temps1.append(X[row][0])
                temps2.append(X[row][1])
        # find the mean of those rows and assign it to C[k]
        if pd.isna(np.mean(temps1)) or pd.isna(np.mean(temps2)):
            C.append([0, 0])
        else:
            C.append([np.mean(temps1), np.mean(temps2)])

    return I, C
    


def compute_objective(X, C, I):
    """ Compute the clustering objective for X and C
    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    C: array, shape(k ,d)
        The final cluster centers

    Returns
    -------
    accuracy: float
        The objective for the given assigments
    """
    # I: nx1 array that tells you which cluster every observation in X belongs to
    obj = 0
    for row in I:
        means = C[row] 
        obj += (means[0] - X[row][0])**2 + (means[1] - X[row][1])**2

    return obj


def k_means_pp(X, k, max_iter):
    """ k-means++ clustering algorithm

    step 1: call k_init() to initialize the centers
    step 2: iteratively refine the assignments

    Parameters
    ----------
    X: array, shape(n ,d)
        Input array of n samples and d features

    k: int
        The number of clusters

    max_iter: int
        Maximum number of iteration

    Returns
    -------
    final_centers: array, shape (k, d)
        The final cluster centers
    objective_values: array, shape (max_iter)
        The objective value at each iteration
    """
    # call k_init() to initialize the centers
        #pass in X and K
        #return C and I-
    I, C = k_init(X, k)
    # objList = []
    iTemp = 0
    #while counter < max_iter and I does not equal iTemp 
    for i in range(max_iter):
        if (I == iTemp):
            break
        iTemp = I
        #call assign_data2clusters 
            #passing in X, C, I, k
            # returns I, C
        I, C, = assign_data2clusters(X, C, I, k)
        # #objList.append(compute_objective())
        # objList.append(compute_objective(X, C, I))
    obj = compute_objective(X, C, I)

    return C, obj, I



data = pd.DataFrame(pd.read_csv('iris.data', names=["sepalLength", "sepalWidth", "petalLength", "petalWidth", "class"]))

sepalRatio = data['sepalLength'] / data['sepalWidth']
petalRatio = data['petalLength'] / data['petalWidth']

data = [sepalRatio]
data.append(petalRatio)

X = np.array(data).T.tolist()



C1, obj1, I1 = k_means_pp(X, 1, 10000)
C2, obj2, I2 = k_means_pp(X, 2, 10000)
C3, obj3, I3 = k_means_pp(X, 3, 10000)
C4, obj4, I4 = k_means_pp(X, 4, 10000)
C5, obj5, I5 = k_means_pp(X, 5, 10000)

obj = [obj1, obj2, obj3, obj4, obj5]

plt.plot([1, 2, 3, 4, 5], obj, 'o')
plt.xlabel("k values")
plt.ylabel("objective")
plt.show()


#Create a plot showing how objective changes with number of iterations.

#For this I will use k = 3.
objs = []
nums = range(100, 1000)
for num in nums:
    C1, obj1, I1 = k_means_pp(X, 3, num)
    objs.append(obj1)

    
# plt.plot(nums, objs, 'o')
# plt.xlabel("number of iterations")
# plt.ylabel("objective")
# plt.show()

# Create a plot with the data colored by assignment, and the cluster centers.

colors = ["red", "yellow", "purple", "pink"]
colormap = matplotlib.colors.ListedColormap(colors)
plt.scatter(sepalRatio, petalRatio, c=I4, cmap=colormap)
plt.show()