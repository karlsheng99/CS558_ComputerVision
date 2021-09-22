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
                    real_i = i - int(kernel_size / 2) + kernel_i
                    real_j = j - int(kernel_size / 2) + kernel_j
                    if real_i < 0:
                        if real_j < 0:
                            intensity = source[0, 0]
                        elif 0 <= real_j < col:
                            intensity = source[0, real_j]
                        elif real_j >= col:
                            intensity = source[0, col - 1]
                    elif 0 <= real_i < row:
                        if real_j < 0:
                            intensity = source[real_i, 0]
                        elif 0 <= real_j < col:
                            intensity = source[real_i, real_j]
                        elif real_j >= col:
                            intensity = source[real_i, col - 1]
                    elif real_i >= row:
                        if real_j < 0:
                            intensity = source[row - 1, 0]
                        elif 0 <= real_j < col:
                            intensity = source[row - 1, real_j]
                        elif real_j >= col:
                            intensity = source[row - 1, col - 1]

                    intensity = np.multiply(intensity, kernel[kernel_i][kernel_j])
                    new_value = np.add(new_value, intensity)
                    #if i==319 and j==0:
                    #    print('\t', real_i, real_j, intensity)
            #print(i, j, new_value)
            new_matrix[i][j] = new_value


    return new_matrix

