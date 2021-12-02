import numpy as np
from itertools import product


def convolve(source, kernel):
    """
    :param source: image pixel matrix
    :param kernel: kernel matrix
    :return: convolved matrix
    """

    row, col = source.shape
    kernel_size = kernel.shape[0]

    pad_row = int(kernel_size / 2)
    pad_col = int(kernel_size / 2)

    pad_matrix = np.zeros((row + (2 * pad_row), col + (2 * pad_col)))

    for i in range(len(pad_matrix)):
        for j in range(len(pad_matrix[0])):
            if i < pad_row:
                if j < pad_col:
                    pad_matrix[i][j] = source[0][0]
                elif pad_col <= j < pad_col + col:
                    pad_matrix[i][j] = source[0][j - pad_col]
                else:
                    pad_matrix[i][j] = source[0][col - 1]
            elif pad_row <= i < pad_row + row:
                if j < pad_col:
                    pad_matrix[i][j] = source[i - pad_row][0]
                elif pad_col <= j < pad_col + col:
                    pad_matrix[i][j] = source[i - pad_row][j - pad_col]
                else:
                    pad_matrix[i][j] = source[i - pad_row][col - 1]
            else:
                if j < pad_col:
                    pad_matrix[i][j] = source[row - 1][0]
                elif pad_col <= j < pad_col + col:
                    pad_matrix[i][j] = source[row - 1][j - pad_col]
                else:
                    pad_matrix[i][j] = source[row - 1][col - 1]

    flattened_image = np.zeros((row * col, kernel_size * kernel_size))

    y = 0
    for i, j in product(range(row), range(col)):
        flattened_image[y, :] = np.ravel(pad_matrix[i: i + kernel_size, j: j + kernel_size])
        y += 1

    flattened_kernel = np.ravel(kernel)

    convolved_matrix = np.dot(flattened_image, flattened_kernel).reshape(row, col)

    return convolved_matrix
