import numpy as np
import math
from gmm import distance_function


def greedy_diverse_mod(d_con, d_cat, max_diff, k, w1, w2):
    top_k = []

    # pick first 2
    distance_matrix = distance_function(d_con, max_diff, d_cat, w1, w2)
    print "found my distance matrix = ", distance_matrix

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
        # sub_matrix_min_val_per_col = np.min(sub_matrix, axis=1)
        sub_matrix_sum_val_per_col = np.sum(sub_matrix, axis=0)
        # print "sub-matrix minimum value per column  = ", sub_matrix_min_val_per_col
        # print "sub-matrix sum value per column  = ", sub_matrix_sum_val_per_col

        # obtain the point and add it to the sample
        # sub_matrix_max_value_row_index = list(sub_matrix_min_val_per_col).index(max(sub_matrix_min_val_per_col))
        sub_matrix_max_sum_value_row_index = list(sub_matrix_sum_val_per_col).index(max(sub_matrix_sum_val_per_col))
        # print "sub-matrix max minimum value row = ", sub_matrix_max_value_row_index
        # print "sub-matrix max sum value row = ", sub_matrix_max_sum_value_row_index

        # next_point_index = list(sub_matrix[sub_matrix_max_value_row_index]).index(max(sub_matrix_min_val_per_col))
        # next_point_index = list(sub_matrix[sub_matrix_max_sum_value_row_index]).index(max(sub_matrix_sum_val_per_col))

        print "next point index = ", sub_matrix_max_sum_value_row_index

        top_k.append(cols[sub_matrix_max_sum_value_row_index])

    print "final set = ", top_k
    # print "final set ordered  = ", sorted(top_k)
    # remember to return

    return top_k

# d = np.asmatrix(np.array([[10], [20], [30], [40], [50], [0]]))
# dcat = np.asmatrix(np.array([[1], [0], [1], [1], [0], [1]]))
#
# print greedy_diverse_mod(d, dcat, 50, 4, 1, 0)
