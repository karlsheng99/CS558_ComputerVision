from pathlib import Path
from PIL import Image
import numpy as np
import convolution
import gaussian
import hessian
import ransac
import hough


def pre_processing(source_matrix, sigma, threshold):
    """
        sigma = 1 -> sigma=1 * 1
        sigma = 2 -> sigma=1 * 3
        sigma = 4 -> sigma=1 * 5
        sigma = 8 -> sigma=1 * 7
        """
    smoothed_matrix = np.copy(source_matrix)
    if sigma == 1 or sigma == 2 or sigma == 4 or sigma == 8:
        gauss_kernel = gaussian.gaussian_filter(1)
        for i in range(int(np.log2(sigma)) * 2 + 1):
            smoothed_matrix = convolution.convolve(smoothed_matrix, gauss_kernel)
    else:
        gauss_kernel = gaussian.gaussian_filter(sigma)
        smoothed_matrix = convolution.convolve(smoothed_matrix, gauss_kernel)

    hessain_matrix = hessian.hessian_detector(smoothed_matrix, threshold)

    return hessain_matrix


def line_detection(sigma=1, threshold=120000, min_inlier=30, dist_threshold=2, num_bin_x=181, num_bin_y=0):
    source_image = Image.open('road.png')
    result_dir = 'results/'
    if not Path(result_dir).exists():
        Path(result_dir).mkdir()
    source_matrix = np.array(source_image)

    hessian_matrix = pre_processing(source_matrix, sigma, threshold)
    hessian_image = Image.fromarray(hessian_matrix)
    hessian_image.convert("L").save(result_dir + 'hessian_sigma=' + str(sigma) + '_threshold=' + str(threshold) + '.png')

    ransac.ransac_findlines(source_image, hessian_matrix, min_inlier, dist_threshold, result_dir)

    num_bin_y = num_bin_y
    min_bin_y = np.sqrt(hessian_matrix.shape[0] ** 2 + hessian_matrix.shape[1] ** 2) * 2
    if num_bin_y < min_bin_y:
        num_bin_y = min_bin_y

    hough_space_matrix = hough.hough_findlines(source_image, hessian_matrix, result_dir, num_bin_x, num_bin_y)
    hough_space_image = Image.fromarray(hough_space_matrix)

    hough_space_image.convert("L").save(result_dir + 'hough_space_bin-x=' + str(int(num_bin_x)) + '_bin-y=' + str(int(num_bin_y)) + '.png')


def main():
    line_detection()


if __name__ == '__main__':
    main()
