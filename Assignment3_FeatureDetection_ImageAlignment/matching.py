import numpy as np


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


def ssd(source_image_1, source_image_2, source_matrix_1, source_matrix_2, harris_matrix_1, harris_matrix_2, window_size=3):
    points_1 = detect_points(harris_matrix_1)
    points_2 = detect_points(harris_matrix_2)
    window_half_size = int(window_size / 2)

    # Keep tracking the coordinates of top 20 matching points
    best_20_matches = []

    # Keep tracking the ssd value of top 20 matching points
    best_20_ssd = np.full(20, np.inf)

    for point_1 in points_1:
        y1, x1 = point_1
        for point_2 in points_2:
            y2, x2 = point_2

            ssd_matrix = np.array((source_matrix_1[y1 - window_half_size: y1 + window_half_size + 1,
                                                   x1 - window_half_size: x1 + window_half_size + 1] -
                                   source_matrix_2[y2 - window_half_size: y2 + window_half_size + 1,
                                                   x2 - window_half_size: x2 + window_half_size + 1]) ** 2)
            ssd_value = ssd_matrix.sum()

            if ssd_value >= best_20_ssd[19]:
                break
            else:
                for i in range(20):
                    if ssd_value < best_20_ssd[i]:
                        best_20_ssd = np.insert(best_20_ssd, i, ssd_value)
                        best_20_ssd = best_20_ssd[:-1]

                        best_20_matches.insert(i, [point_1, point_2])
                        if len(best_20_matches) > 20:
                            best_20_matches = best_20_matches[:-1]

                        break

    return best_20_matches



