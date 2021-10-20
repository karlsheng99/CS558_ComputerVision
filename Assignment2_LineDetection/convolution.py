import numpy as np
import math
from itertools import product
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros


def convolve2(source, kernel):
    """
    :param source: image pixel matrix
    :param kernel: kernel matrix
    :return: convolved matrix
    """

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


def convolve(image, filter):
    # image: image pixel matrix
    # filter: filter matrix
    # Output is the concolved image

    row, col = image.shape
    kernel_size = filter.shape[0]
    convolved_image = np.zeros((row, col))

    pad_row = int((kernel_size - 1)/2)
    pad_col = int((kernel_size - 1)/2)

    result = np.zeros((row + (2 * pad_row), col + (2 * pad_col)))

    # copy image into center of result image
    # result[pad_row:pad_row + row, pad_col:pad_col + col] = image

    for i in range(len(result)):
        for j in range(len(result[0])):
            if i < pad_row:
                if j < pad_col:
                    result[i][j] = image[0][0]
                elif pad_col <= j < pad_col + col:
                    result[i][j] = image[0][j - pad_col]
                else:
                    result[i][j] = image[0][col - 1]
            elif pad_row <= i < pad_row + row:
                if j < pad_col:
                    result[i][j] = image[i - pad_row][0]
                elif pad_col <= j < pad_col + col:
                    result[i][j] = image[i - pad_row][j - pad_col]
                else:
                    result[i][j] = image[i - pad_row][col - 1]
            else:
                if j < pad_col:
                    result[i][j] = image[row - 1][0]
                elif pad_col <= j < pad_col + col:
                    result[i][j] = image[row - 1][j - pad_col]
                else:
                    result[i][j] = image[row - 1][col - 1]


    # Result image dimension
    R_height, R_width = result.shape[0], result.shape[1]
    flt_height = R_height - kernel_size + 1
    flt_width = R_width - kernel_size + 1

    # im2col, turn the kernel_size*kernel_size pixels into a row and np.vstack all rows
    image_array = np.zeros((flt_height * flt_width, kernel_size * kernel_size))
    row = 0
    for i, j in product(range(flt_height), range(flt_width)):
        window = ravel(result[i : i + kernel_size, j : j + kernel_size])  # Flatten the window matrix to an array
        image_array[row, :] = window
        row += 1

    #  turn the kernel into shape(k*k, 1)
    filter_array = ravel(filter)

    # reshape and get the filtered image
    flt = dot(image_array, filter_array).reshape(flt_height, flt_width)

    return flt
