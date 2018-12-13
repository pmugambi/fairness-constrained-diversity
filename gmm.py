import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

data = np.random.rand(150, 2)  # 100 rows of 2 x,y values
print "data = ", data


def dist_matrix(points):
    dist_mat = squareform(pdist(points, metric="euclidean"))

    # mask the diagonal
    np.fill_diagonal(dist_mat, np.nan)
    print "dist mat = ", dist_mat

    # and calculate the minimum of each row (or column)
    min_distances = np.nanmin(dist_mat, axis=1)
    print "res = ", min_distances

    # print "distances = ", res
    return min_distances, dist_mat


top_k = []


def greedy_diverse(d, k):
    # pick first 2
    min_distances, distance_matrix = dist_matrix(d)

    # max(min_distances)
    max_min_index = list(min_distances).index(max(min_distances))
    max_min_value = max(min_distances)

    print max_min_value

    # point 1's index
    print max_min_index

    top_k.append(max_min_index)

    point2_index = np.argwhere(distance_matrix[max_min_index,:]==max_min_value)[0][0]
    print point2_index

    top_k.append(point2_index)

    while len(top_k) < k:
        # select matrix region of interest
        rows = []
        cols = []
        for i in xrange(0, data.shape[0]):
            if i in top_k:
                rows.append(i)
            else:
                cols.append(i)
        print "rows = ", rows, " and cols = ", cols

        # obtain pairwise distances for region of interest
        sub_matrix = distance_matrix[np.ix_(rows, cols)]
        print "section of interest = ", sub_matrix

        # obtain min distance for each point, with respect to the chosen points
        sub_max_min_dist_per_col = np.min(sub_matrix, axis=1)
        print "res = ", sub_max_min_dist_per_col

        # obtain the point and add it to the sample
        sub_max_min_index = list(sub_max_min_dist_per_col).index(max(sub_max_min_dist_per_col))
        print "sub mat max = ", sub_max_min_index
        top_k.append(cols[sub_max_min_index])

    print "final set = ", top_k
    print "final set ordered  = ", sorted(top_k)

    # remember to return

greedy_diverse(data, 60)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

x = data[:,0]
y = data[:,1]

colour = []
for i in xrange(0, data.shape[0]):
    if i in top_k:
        colour.append("green")
    else:
        colour.append("blue")

ax.scatter(x, y, color=colour)
# for i, txt in enumerate(colour):
#     ax.annotate(i, (x[i], y[i]))

plt.show()
