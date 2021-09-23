import numpy as np


def convolve(source, kernel):
    row, col = source.shape
    kernel_size = kernel.shape[0]
    new_matrix = np.zeros((row, col))

    for i in range(row):
        for j in range(col):
            new_value = 0
            for kernel_i in range(kernel_size):
                for kernel_j in range(kernel_size):
                    source_i = i - int(kernel_size / 2) + kernel_i
                    source_j = j - int(kernel_size / 2) + kernel_j

                    if source_i < 0:
                        source_i = 0
                    elif source_i >= row:
                        source_i = row - 1

                    if source_j < 0:
                        source_j = 0
                    elif source_j >= col:
                        source_j = col - 1

                    intensity = source[source_i][source_j]
                    intensity = np.multiply(intensity, kernel[kernel_i][kernel_j])
                    new_value = np.add(new_value, intensity)

            new_matrix[i][j] = new_value

    return new_matrix

