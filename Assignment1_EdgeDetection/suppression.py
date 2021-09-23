import gradient
import numpy as np

gradient_magnitude = np.array([[]])
gradient_direction = np.array([[]])
row = 0
col = 0


def find_edge(i, j, i1, j1, i2, j2):
    """
    :param i: center v-coordinate
    :param j: center h-coordinate
    :param i1: neighbor 1 v-coordinate
    :param j1: neighbor 1 h-coordinate
    :param i2: neighbor 2 v-coordinate
    :param j2: neighbor 2 h-coordinate
    :return: edge value
    """

    # Neighbor 1 does not exist
    if i1 < 0 or i1 >= row or j1 < 0 or j1 >= col:
        # No neighbor exists
        if i2 < 0 or i2 >= row or j2 < 0 or j2 >= col:
            return gradient_magnitude[i][j]
        # Only neighbor 2 exists
        if gradient_magnitude[i][j] < gradient_magnitude[i2][j2]:
            return 0
        else:
            return gradient_magnitude[i][j]
    # Neighbor 2 does not exist
    if i2 < 0 or i2 >= row or j2 < 0 or j2 >= col:
        # Only neighbor 1 exists
        if gradient_magnitude[i][j] < gradient_magnitude[i1][j1]:
            return 0
        else:
            return gradient_magnitude[i][j]
    # Both neighbors exist
    if gradient_magnitude[i][j] < gradient_magnitude[i1][j1] or gradient_magnitude[i][j] < gradient_magnitude[i2][j2]:
        return 0
    else:
        return gradient_magnitude[i][j]


def find_orientation(i, j, theta):
    """
    :param i: v-coordinate
    :param j: h-coordinate
    :param theta: gradient orientation
    :return: edge value
    """

    if theta >= 67.5 or theta < -67.5:
        return find_edge(i, j, i - 1, j, i + 1, j)
    elif 22.5 <= theta < 67.5:
        return find_edge(i, j, i - 1, j + 1, i - 1, j + 1)
    elif -22.5 <= theta < 22.5:
        return find_edge(i, j, i, j - 1,  i, j + 1)
    elif -67.6 <= theta < -22.5:
        return find_edge(i, j, i + 1, j + 1, i + 1, j + 1)


def non_max_suppress(gradient_mag, gradient_dir):
    """
    :param gradient_mag: gradient magnitude from gradient computation
    :param gradient_dir: gradient direction from gradient computation
    :return: non-maximum suppressed matrix
    """

    global row, col, gradient_direction, gradient_magnitude
    gradient_magnitude = gradient_mag
    gradient_direction = gradient_dir
    row, col = gradient_magnitude.shape
    suppressed_matrix = np.zeros((row, col))

    for i in range(row):
        for j in range(col):
            suppressed_matrix[i][j] = find_orientation(i, j, gradient_direction[i][j])

    return suppressed_matrix

