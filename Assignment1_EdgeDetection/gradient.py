import numpy as np
import convolution


def compute_gradient(source, threshold=0):
    """
    :param source: image pixel matrix
    :param threshold: pixel intensity (default=0)
    :return: gradient_magnitude matrix & gradient_direction matrix
    """

    # Sobel filter
    v_kernel = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])

    h_kernel = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])

    v_new_matrix = np.array(convolution.convolve(source, v_kernel))
    h_new_matrix = np.array(convolution.convolve(source, h_kernel))

    gradient_magnitude = np.sqrt(np.square(v_new_matrix) + np.square(h_new_matrix))
    gradient_magnitude = np.where(gradient_magnitude < threshold, 0, gradient_magnitude)
    gradient_direction = np.rad2deg(np.arctan(h_new_matrix / v_new_matrix))

    return gradient_magnitude, gradient_direction

