import numpy as np


def detect_points(source):
    row, col = source.shape
    points = []
    for i in range(row):
        for j in range(col):
            if source[i][j] != 0:
                points.append((i, j))

    return points


def hough_transform(hessian_matrix):
    y, x = hessian_matrix.shape
    bin_x = 181
    bin_y = np.sqrt(x * x + y * y) * 2
    accumulator = np.zeros((int(bin_y), bin_x))

    points = detect_points(hessian_matrix)

    for point in points:
        for theta in range(bin_x):
            rho = point[1] * np.cos(np.deg2rad(theta)) + point[0] * np.sin(np.deg2rad(theta))
            accumulator[int(rho)][theta] += 1

    return accumulator*100