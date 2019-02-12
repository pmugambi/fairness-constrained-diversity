import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, euclidean

data = np.random.rand(3, 1)  # 100 rows of 2 x,y values
# print "data = ", data


def similarity_function(cat_data):
    # if u == v:
    #     return 0
    # else:
    #     return 1
    print "cat_data received = ", cat_data
    print "cosine distances = ", squareform(pdist(cat_data, metric="hamming"))
    return squareform(pdist(cat_data, metric="hamming"))


def euclidean_normed_function(con_data, max_diff):
    print "con_data received = ", con_data
    print "euclidean distances = ", squareform(pdist(con_data, metric="euclidean")) / max_diff
    return squareform(pdist(con_data, metric="euclidean")) / max_diff


def distance_function(con_data, max_diff, cat_data):
    dist_mat = 0.5 * (euclidean_normed_function(con_data, max_diff) + similarity_function(cat_data))
    np.fill_diagonal(dist_mat, np.nan)
    min_distances = np.nanmin(dist_mat, axis=1)
    return min_distances, dist_mat


def dist_matrix(points):
    # print "received points = ", points
    dist_mat = squareform(pdist(points, metric="euclidean"))
    print "dist_mat = ", dist_mat

    # mask the diagonal
    np.fill_diagonal(dist_mat, np.nan)
    # print "dist mat = ", dist_mat

    # and calculate the minimum of each row (or column)
    min_distances = np.nanmin(dist_mat, axis=1)
    # print "min distances = ", min_distances

    # print "distances = ", res
    return min_distances, dist_mat


top_k = []


def greedy_diverse(d, k):
    # pick first 2
    min_distances, distance_matrix = dist_matrix(d)

    # max(min_distances)
    max_min_index = list(min_distances).index(max(min_distances))
    max_min_value = max(min_distances)

    # print max_min_value

    # point 1's index
    print "point 1 index = ", max_min_index

    top_k.append(max_min_index)

    point2_index = np.argwhere(distance_matrix[max_min_index, :] == max_min_value)[0][0]
    print "point 2 index = ", point2_index

    top_k.append(point2_index)

    while len(top_k) < k:
        # select matrix region of interest
        rows = []
        cols = []
        for i in xrange(0, d.shape[0]):
            if i in top_k:
                rows.append(i)
            else:
                cols.append(i)
        # print "rows = ", rows, " and cols = ", cols

        # obtain pairwise distances for region of interest
        sub_matrix = distance_matrix[np.ix_(rows, cols)]
        # print "section of interest = ", sub_matrix

        # obtain min distance for each point, with respect to the chosen points
        sub_max_min_dist_per_col = np.min(sub_matrix, axis=1)
        # print "res = ", sub_max_min_dist_per_col

        # obtain the point and add it to the sample
        sub_max_min_index = list(sub_max_min_dist_per_col).index(max(sub_max_min_dist_per_col))
        # print "sub mat max = ", sub_max_min_index
        top_k.append(cols[sub_max_min_index])

    print "final set = ", top_k
    print "final set ordered  = ", sorted(top_k)
    # remember to return

    return top_k


top_k2 = []


def greedy_diverse_mod(d_con, d_cat, max_diff, k):
    # pick first 2
    min_distances, distance_matrix = distance_function(d_con, max_diff, d_cat)

    print "found my distance matrix = ", distance_matrix

    # max(min_distances)
    max_min_index = list(min_distances).index(max(min_distances))
    max_min_value = max(min_distances)

    # print max_min_value

    # point 1's index
    print "point 1 index = ", max_min_index

    top_k2.append(max_min_index)

    point2_index = np.argwhere(distance_matrix[max_min_index, :] == max_min_value)[0][0]
    print "point 2 index = ", point2_index

    top_k2.append(point2_index)

    while len(top_k2) < k:
        # select matrix region of interest
        rows = []
        cols = []
        for i in xrange(0, d_con.shape[0]):
            if i in top_k2:
                rows.append(i)
            else:
                cols.append(i)
        # print "rows = ", rows, " and cols = ", cols

        # obtain pairwise distances for region of interest
        sub_matrix = distance_matrix[np.ix_(rows, cols)]
        print "section of interest = ", sub_matrix

        # obtain min distance for each point, with respect to the chosen points
        sub_max_min_dist_per_col = np.min(sub_matrix, axis=1)
        print "res = ", sub_max_min_dist_per_col

        # obtain the point and add it to the sample
        sub_max_min_index = list(sub_max_min_dist_per_col).index(max(sub_max_min_dist_per_col))
        # print "sub mat max = ", sub_max_min_index
        top_k2.append(cols[sub_max_min_index])

    print "final set = ", top_k2
    print "final set ordered  = ", sorted(top_k2)
    # remember to return

    return top_k2


# dist_matrix(data)

# print "final dist = ", distance_function(np.asmatrix(np.array([[50], [0], [30]])), 50, np.asmatrix(np.array([[0], [1], [0]])))

# greedy_diverse(data, 5)
#
#
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
#
# x = data[:,0]
# y = data[:,1]
#
# colour = []
# for i in xrange(0, data.shape[0]):
#     if i in top_k:
#         colour.append("green")
#     else:
#         colour.append("blue")
#
# ax.scatter(x, y, color=colour)
# # for i, txt in enumerate(colour):
# #     ax.annotate(i, (x[i], y[i]))
#
# plt.show()
