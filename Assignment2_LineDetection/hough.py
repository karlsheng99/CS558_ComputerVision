import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def detect_points(source):
    """
    :param source: hessian matrix
    :return: list of coordinates of points with intensity which is not equal to 0
    """

    row, col = source.shape
    points = []
    for i in range(row):
        for j in range(col):
            if source[i][j] != 0:
                points.append((i, j))

    return points


def find_maxima(accumulator):
    """
    :param accumulator: hough space accumulator matrix
    :return: list of the polar coordinates of 4 points with the highest votes
    """

    y, x = accumulator.shape
    point_1 = (0, 0)
    point_2 = (0, 0)
    point_3 = (0, 0)
    point_4 = (0, 0)

    for rho in range(y):
        for theta in range(x):
            if accumulator[rho][theta] > accumulator[point_1[0]][point_1[1]]:
                point_4 = point_3
                point_3 = point_2
                point_2 = point_1
                point_1 = (rho, theta)
            elif accumulator[rho][theta] > accumulator[point_2[0]][point_2[1]]:
                point_4 = point_3
                point_3 = point_2
                point_2 = (rho, theta)
            elif accumulator[rho][theta] > accumulator[point_3[0]][point_3[1]]:
                point_4 = point_3
                point_3 = (rho, theta)
            elif accumulator[rho][theta] > accumulator[point_4[0]][point_4[1]]:
                point_4 = (rho, theta)

    return [point_1, point_2, point_3, point_4]


def plot_lines(image, points, polar_coord, result_dir):
    """
    :param image: source image
    :param points: list of coordinates of points with non-zero intensity
    :param polar_coord: list of the polar coordinates of 4 points with highest votes
    :param result_dir: plot the points and lines on the image and save it to this directory
    """

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for line in polar_coord:
        rho, theta = line
        left_most = points[0]
        right_most = points[0]

        for point in points:
            y, x = point
            if int(x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta))) == rho:
                rect = patches.Rectangle((x, y), 3, 3, facecolor='r')
                ax.add_patch(rect)

                if x < left_most[1]:
                    left_most = point
                elif x > right_most[1]:
                    right_most = point

        plt.plot([left_most[1], right_most[1]], [left_most[0], right_most[0]], color='red', linewidth=1)

    plt.show()
    fig.savefig(result_dir + 'hough.png')


def hough_findlines(image, hessian_matrix, result_dir, num_bin_x=181, num_bin_y=0):
    """
    :param image: source image
    :param hessian_matrix: hessian matrix
    :param result_dir: save file to this directory
    :param num_bin_x: dimension of bin_x of the accumulator, default = 181 (0-180 degree)
    :param num_bin_y: dimension of bin_y of the accumulator, default = 2*sqrt(row^2+col^2) (maximum range of rho)
    :return: hough space accumulator matrix * 100
    """

    y, x = hessian_matrix.shape
    bin_x = num_bin_x
    bin_y = num_bin_y
    if num_bin_y == 0:
        bin_y = np.sqrt(x * x + y * y) * 2
    accumulator = np.zeros((int(bin_y), bin_x))

    points = detect_points(hessian_matrix)

    for point in points:
        for theta in range(bin_x):
            rho = point[1] * np.cos(np.deg2rad(theta)) + point[0] * np.sin(np.deg2rad(theta))
            accumulator[int(rho)][theta] += 1

    best_lines_polar = find_maxima(accumulator)
    plot_lines(image, points, best_lines_polar, result_dir)

    return accumulator * 100

