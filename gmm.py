import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, euclidean
import math

data = np.random.rand(3, 1)  # 100 rows of 2 x,y values
# print "data = ", data


def similarity_function(cat_data):
    return squareform(pdist(cat_data, metric="hamming"))


def euclidean_normed_function(con_data, max_diff):
    return squareform(pdist(con_data, metric="euclidean")) / max_diff


def distance_function(con_data, max_diff, cat_data, w1, w2):
    # dist_mat = 0.5 * (euclidean_normed_function(con_data, max_diff) + similarity_function(cat_data)) # original
    # function
    dist_mat = (w1 * euclidean_normed_function(con_data, max_diff)) + (w2 * similarity_function(cat_data))
    return dist_mat


def greedy_diverse_mod(d_con, d_cat, max_diff, k, w1, w2):
    top_k = []

    # print "received d_con = ", d_con
    # print "received d_cat = ", d_cat

    # pick first 2
    distance_matrix = distance_function(d_con, max_diff, d_cat, w1, w2)
    # print "found my distance matrix = ", distance_matrix

    max_distance = np.max(distance_matrix)
    # print "max val  = ", max_distance

    max_value_index = np.argmax(distance_matrix)

    # print "max flat array index = ", max_value_index

    max_value_index_on_mat = math.floor(max_value_index/float(distance_matrix.shape[0]))
    print "point 1 index = ", max_value_index_on_mat

    # print "testing ceiling ", math.floor(max_value_index/float(distance_matrix.shape[0]))

    top_k.append(int(max_value_index_on_mat))

    point2_index = np.argwhere(distance_matrix[max_value_index_on_mat, :] == max_distance)[0][0]
    print "point 2 index = ", point2_index
    top_k.append(point2_index)

    while len(top_k) < k:
        # select matrix region of interest
        rows = []
        cols = []
        for i in xrange(0, d_con.shape[0]):
            if i in top_k:
                rows.append(i)
            else:
                cols.append(i)
        # print "rows = ", rows, " and cols = ", cols

        # obtain pairwise distances for region of interest
        sub_matrix = distance_matrix[np.ix_(rows, cols)]
        # print "section of interest = ", sub_matrix

        # obtain min distance for each point, with respect to the chosen points
        sub_matrix_min_val_per_col = np.min(sub_matrix, axis=1)
        # print "sub-matrix minimum value per column  = ", sub_matrix_min_val_per_col

        # obtain the point and add it to the sample
        sub_matrix_max_value_row_index = list(sub_matrix_min_val_per_col).index(max(sub_matrix_min_val_per_col))
        # print "sub-matrix max minimum value row = ", sub_matrix_max_value_row_index

        next_point_index = list(sub_matrix[sub_matrix_max_value_row_index]).index(max(sub_matrix_min_val_per_col))

        top_k.append(cols[next_point_index])

    print "final set = ", top_k
    # print "final set ordered  = ", sorted(top_k)
    return top_k


# dist_matrix(data)

# d = np.asmatrix(np.array([[10], [20], [30], [40], [50], [0]]))
# d = np.asmatrix(np.array([[0], [52], [0], [20], [0], [0]]))
# d = np.asmatrix(np.array([[0.0], [52.0], [0.0], [0.0], [0.0], [52.0], [52.0], [30.0], [52.0], [52.0]]))
#
# dcat = np.asmatrix(np.array([[1], [0], [1], [1], [0], [1], [0], [1], [1], [0]]))
# print greedy_diverse_mod(d, dcat, 52, 3, 1, 0)

# print "final dist = ", distance_function(np.asmatrix(np.array([[50], [0], [30]])), 50, np.asmatrix(np.array([[0], [1], [0]])))

# print greedy_diverse_2(d, 3)
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
