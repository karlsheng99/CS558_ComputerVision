from sympy import Symbol, solve
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
    x_l_1, y_l_1 = three_pairs[0][0]
    x_l_2, y_l_2 = three_pairs[1][0]
    x_l_3, y_l_3 = three_pairs[2][0]
    x_r_1, y_r_1 = three_pairs[0][1]
    x_r_2, y_r_2 = three_pairs[1][1]
    x_r_3, y_r_3 = three_pairs[2][1]

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

    affine_matrix = [float(solution[a]), float(solution[b]), float(solution[c]), float(solution[d]), float(solution[e]), float(solution[f])]

    return affine_matrix


def find_residual(affine_matrix, pair):
    x_l, y_l = pair[0]
    x_r, y_r = pair[1]

    d_x = x_l - (affine_matrix[0] * x_r + affine_matrix[1] * y_r + affine_matrix[2])
    d_y = y_l - (affine_matrix[3] * x_r + affine_matrix[4] * y_r + affine_matrix[5])

    print(float(abs(d_x) + abs(d_y)))

    return abs(d_x) + abs(d_y)


def ransac(correspondences, min_inlier, dist_threshold):
    # Initial number of points
    s = 3
    # Distance threshold
    t = dist_threshold
    # Outlier ratio
    e = 0.5
    # Probability of a least one random sample is free from outliers
    p = 0.99
    # Number of samples
    n = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** s)))

    best_affine_matrix = []
    best_num_inlier = 0

    for i in range(n):
        inlier_subset = random.sample(correspondences, 3)
        affine_matrix = find_affine_matrix(inlier_subset)

        num_inlier = 0

        for pair in correspondences:
            if find_residual(affine_matrix, pair) <= dist_threshold:
                num_inlier += 1

        print(i, num_inlier)
        print(affine_matrix)

        if num_inlier > best_num_inlier:
            best_num_inlier = num_inlier
            best_affine_matrix.clear()
            best_affine_matrix = affine_matrix.copy()

        if num_inlier > min_inlier:
            break

    return best_affine_matrix


def affine_transform(harris_matrix_1, harris_matrix_2, best_20_pairs):
    random_30_pairs = select_random_correspondences(harris_matrix_1, harris_matrix_2)

    affine_matrix = ransac(best_20_pairs, 13, 70)

    return affine_matrix

