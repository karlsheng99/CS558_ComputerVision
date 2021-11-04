from sympy import Symbol, solve
from scipy import ndimage as ndi
import numpy as np
import random


def detect_points(source):
    """
    :param source: hessian matrix
    :return: list of coordinates of points with intensity which is not equal to 0
    """

    row, col = source.shape
    points = []
    for i in range(row):
        for j in range(col):
            if source[i][j] != 0:
                points.append((i, j))

    return points


def select_random_correspondences(harris_matrix_1, harris_matrix_2):
    points_1 = detect_points(harris_matrix_1)
    points_2 = detect_points(harris_matrix_2)

    random_list_1 = random.sample(points_1, 30)
    random_list_2 = random.sample(points_2, 30)

    random_30_pairs = []
    for i in range(len(random_list_1)):
        random_30_pairs.append([random_list_1[i], random_list_2[i]])

    return random_30_pairs


def find_affine_matrix(three_pairs):
    y_l_1, x_l_1 = three_pairs[0][0]
    y_l_2, x_l_2 = three_pairs[1][0]
    y_l_3, x_l_3 = three_pairs[2][0]
    y_r_1, x_r_1 = three_pairs[0][1]
    y_r_2, x_r_2 = three_pairs[1][1]
    y_r_3, x_r_3 = three_pairs[2][1]

    a = Symbol('a')
    b = Symbol('b')
    c = Symbol('c')
    d = Symbol('d')
    e = Symbol('e')
    f = Symbol('f')

    equations = [a * x_r_1 + b * y_r_1 + c - x_l_1,
                 d * x_r_1 + e * y_r_1 + f - y_l_1,
                 a * x_r_2 + b * y_r_2 + c - x_l_2,
                 d * x_r_2 + e * y_r_2 + f - y_l_2,
                 a * x_r_3 + b * y_r_3 + c - x_l_3,
                 d * x_r_3 + e * y_r_3 + f - y_l_3]

    unknowns = [a, b, c, d, e, f]

    solution = solve(equations, unknowns)

    try:
        affine_matrix = [float(solution[a]), float(solution[b]), float(solution[c]), float(solution[d]), float(solution[e]), float(solution[f])]
    except:
        affine_matrix = [1, 0, 0, 0, 1, 0]

    return affine_matrix


def find_residual(affine_matrix, pair):
    y_l, x_l = pair[0]
    y_r, x_r = pair[1]

    d_x = x_l - (affine_matrix[0] * x_r + affine_matrix[1] * y_r + affine_matrix[2])
    d_y = y_l - (affine_matrix[3] * x_r + affine_matrix[4] * y_r + affine_matrix[5])

    return d_x ** 2 + d_y ** 2


def ransac(best_20_pairs, random_30_pairs, min_inlier, dist_threshold):
    # Initial number of points
    s = 3
    # Distance threshold
    t = dist_threshold
    # Outlier ratio
    e = 0.7
    # Probability of a least one random sample is free from outliers
    p = 0.99
    # Number of samples
    n = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** s)))

    best_affine_matrix = []
    best_num_inlier = 0

    mixed_pairs = np.append(best_20_pairs, random_30_pairs, axis=0)

    for i in range(n):
        inlier_subset = random.sample(best_20_pairs, 3)
        affine_matrix = find_affine_matrix(inlier_subset)

        num_inlier = 0

        for pair in mixed_pairs:
            if find_residual(affine_matrix, pair) <= dist_threshold:
                num_inlier += 1

        if num_inlier > best_num_inlier:
            best_num_inlier = num_inlier
            best_affine_matrix.clear()
            best_affine_matrix = affine_matrix.copy()

        if num_inlier > min_inlier:
            break

    return best_affine_matrix


def affine_transform(harris_matrix_1, harris_matrix_2, best_20_pairs, min_inlier=13, threshold=25):
    random_30_pairs = select_random_correspondences(harris_matrix_1, harris_matrix_2)

    affine_vector = ransac(best_20_pairs, random_30_pairs, min_inlier, threshold)

    return affine_vector


def find_shape_offset(matrix_l, matrix_r, affine_matrix):
    row_l, col_l = matrix_l.shape
    corners_l = np.array([[0, 0], [0, row_l], [col_l, 0], [col_l, row_l]])

    row_r, col_r = matrix_r.shape
    corners_r = np.array([[0, 0], [0, row_r], [col_r, 0], [col_r, row_r]])
    warped_corners = corners_r.dot(affine_matrix[:2, :2]) + affine_matrix[:2, 2]

    all_corners = np.append(corners_l, warped_corners, axis=0)

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    output_shape = np.ceil(corner_max - corner_min)
    output_shape = [int(output_shape[1]), int(output_shape[0])]

    offset = [-int(warped_corners[0][1]), -int(warped_corners[0][0])]

    return output_shape, offset


def panorama_stitching(source_matrix_1, source_matrix_2, affine_vector):
    affine_matrix = np.array([[affine_vector[0], affine_vector[1], affine_vector[2]],
                              [affine_vector[3], affine_vector[4], affine_vector[5]],
                              [0, 0, 1]])

    shape, offset = find_shape_offset(source_matrix_1, source_matrix_2, affine_matrix)

    trans_matrix_1 = ndi.affine_transform(source_matrix_1, np.eye(2), output_shape=shape, cval=-1)
    affine_inv_matrix = np.linalg.inv(affine_matrix)
    trans_matrix_2 = ndi.affine_transform(source_matrix_2, affine_inv_matrix.T[:2, :2], affine_inv_matrix.T[:2, 2] + offset, output_shape=shape, cval=-1)

    mask_1 = (trans_matrix_1 != 0) * 1

    mask_2 = (trans_matrix_2 != 0) * 1

    merged_matrix = np.zeros(shape)
    merged_matrix += trans_matrix_1
    merged_matrix += trans_matrix_2
    overlap = mask_1 * 1.0 + mask_2
    merged_matrix = merged_matrix / np.maximum(overlap, 1)

    return merged_matrix

