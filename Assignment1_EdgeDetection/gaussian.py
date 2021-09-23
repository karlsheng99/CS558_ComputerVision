import numpy as np


def gaussian_filter(sigma):
    size = 6 * sigma + 1

    kernel = np.zeros((size, size))
    for i in range(size):
        row = int(size / 2) - i
        for j in range(size):
            col = int(size / 2) - j
            kernel[i][j] = 1 / (2 * np.pi * sigma * sigma) * np.exp(-(col * col + row * row) / (2 * sigma * sigma))

    const = 1 / np.sum(kernel)
    kernel = np.multiply(kernel, const)

    return kernel

