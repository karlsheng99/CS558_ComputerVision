import numpy as np
import convolution


def non_max_suppress(det_matrix):
    row, col = det_matrix.shape
    suppressed_matrix = np.zeros((row, col))

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            n_11 = det_matrix[i - 1][j - 1]
            n_12 = det_matrix[i - 1][j]
            n_13 = det_matrix[i - 1][j + 1]
            n_21 = det_matrix[i][j - 1]
            n_22 = det_matrix[i][j]
            n_23 = det_matrix[i][j + 1]
            n_31 = det_matrix[i][j - 1]
            n_32 = det_matrix[i + 1][j]
            n_33 = det_matrix[i + 1][j + 1]

            if n_22 == max(n_11, n_12, n_13, n_21, n_22, n_23, n_31, n_32, n_33):
                suppressed_matrix[i][j] = n_22

    return suppressed_matrix


def hessian_detector(source, threshold):
    # Sobel filter
    v_kernel = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    h_kernel = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])

    Ix = np.array(convolution.convolve(source, v_kernel))
    Iy = np.array(convolution.convolve(source, h_kernel))

    Ixx = np.array(convolution.convolve(Ix, v_kernel))
    Ixy = np.array(convolution.convolve(Ix, h_kernel))
    Iyy = np.array(convolution.convolve(Iy, h_kernel))

    det = Ixx * Iyy - Ixy * Ixy

    # det_min = np.min(det)
    # det_max = np.max(det)
    # o_range = det_max - det_min
    # det = (det - det_min) * 255 / o_range

    det = np.where(det < threshold, 0, det)

    sup_det_mat = non_max_suppress(det)

    return sup_det_mat


