import numpy as np
import convolution
import gaussian


def non_max_suppress(source_matrix):
    """
    :param source_matrix: determinant of hessian matrix
    :return: suppressed matrix
    """
    row, col = source_matrix.shape
    suppressed_matrix = np.zeros((row, col))

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            n_11 = source_matrix[i - 1][j - 1]
            n_12 = source_matrix[i - 1][j]
            n_13 = source_matrix[i - 1][j + 1]
            n_21 = source_matrix[i][j - 1]
            n_22 = source_matrix[i][j]
            n_23 = source_matrix[i][j + 1]
            n_31 = source_matrix[i][j - 1]
            n_32 = source_matrix[i + 1][j]
            n_33 = source_matrix[i + 1][j + 1]

            if n_22 == max(n_11, n_12, n_13, n_21, n_22, n_23, n_31, n_32, n_33):
                suppressed_matrix[i][j] = n_22

    return suppressed_matrix


def find_1000_points(source_matrix):
    sorted_points = np.sort(source_matrix.flatten())
    threshold = sorted_points[-1000]

    strongest_points_matrix = np.copy(source_matrix)
    strongest_points_matrix = np.where(strongest_points_matrix < threshold, 0, strongest_points_matrix)

    return strongest_points_matrix


def harris_detector(source_matrix, sigma=1, threshold_R=0):
    row, col = source_matrix.shape
    gauss_kernel = gaussian.gaussian_filter(sigma)
    window_half_size = int(len(gauss_kernel) / 2)

    # Sobel filter
    v_kernel = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    h_kernel = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])

    Ix = np.array(convolution.convolve(source_matrix, v_kernel))
    Iy = np.array(convolution.convolve(source_matrix, h_kernel))

    Ix2 = Ix ** 2
    Iy2 = Iy ** 2
    IxIy = Ix * Iy

    R = np.zeros((row, col))

    for i in range(window_half_size, row - window_half_size):
        for j in range(window_half_size, col - window_half_size):
            Ix2_window = np.array(Ix2[i - window_half_size: i + window_half_size + 1,
                                      j - window_half_size: j + window_half_size + 1]) * gauss_kernel
            Iy2_window = np.array(Iy2[i - window_half_size: i + window_half_size + 1,
                                      j - window_half_size: j + window_half_size + 1]) * gauss_kernel
            IxIy_window = np.array(IxIy[i - window_half_size: i + window_half_size + 1,
                                        j - window_half_size: j + window_half_size + 1]) * gauss_kernel

            M = np.array([[Ix2_window.sum(), IxIy_window.sum()],
                          [IxIy_window.sum(), Iy2_window.sum()]])

            R[i][j] = (M[0][0] * M[1][1] - M[0][1] * M[1][0]) - 0.05 * (M[0][0] + M[1][1]) ** 2
            if R[i][j] <= threshold_R:
                R[i][j] = 0

    # suppressed_R_mat = non_max_suppress(R)

    strongest_1000_mat = find_1000_points(R)
    suppressed_R_mat = non_max_suppress(strongest_1000_mat)

    return suppressed_R_mat
