import numpy as np
import convolution


def compute_gradient(source_matrix):
    """
    :param source_matrix: image pixel matrix
    :param threshold: pixel intensity (default=0)
    :return: gradient_magnitude matrix & gradient_direction matrix
    """

    row, col, rgb = source_matrix.shape

    r_channel = np.empty((row, col))
    g_channel = np.empty((row, col))
    b_channel = np.empty((row, col))

    for i in range(row):
        for j in range(col):
            r_channel[i][j] = source_matrix[i][j][0]
            g_channel[i][j] = source_matrix[i][j][1]
            b_channel[i][j] = source_matrix[i][j][2]

    # Sobel filter
    v_kernel = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    h_kernel = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])

    v_r_channel = np.array(convolution.convolve(r_channel, v_kernel))
    h_r_channel = np.array(convolution.convolve(r_channel, h_kernel))
    v_g_channel = np.array(convolution.convolve(g_channel, v_kernel))
    h_g_channel = np.array(convolution.convolve(g_channel, h_kernel))
    v_b_channel = np.array(convolution.convolve(b_channel, v_kernel))
    h_b_channel = np.array(convolution.convolve(b_channel, h_kernel))

    r_gradient = np.sqrt(np.square(v_r_channel) + np.square(h_r_channel))
    g_gradient = np.sqrt(np.square(v_g_channel) + np.square(h_g_channel))
    b_gradient = np.sqrt(np.square(v_b_channel) + np.square(h_b_channel))

    combined_gradient = np.sqrt(np.square(r_gradient) + np.square(g_gradient) + np.square(b_gradient))

    return combined_gradient

