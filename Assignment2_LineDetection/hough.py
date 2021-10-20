import numpy as np
import matplotlib.pyplot as plt


def detect_points(source):
    row, col = source.shape
    points = []
    for i in range(row):
        for j in range(col):
            if source[i][j] != 0:
                points.append((i, j))

    return points


def find_maxima(accumulator):
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


def plot_lines(image, size, polar_coord, result_dir):
    max_y, max_x = size

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for line in polar_coord:
        rho, theta = line
        x_i = 0
        y_i = int(rho / np.sin(np.deg2rad(theta)))

        if y_i < 0:
            y_i = 0
            x_i = int(rho / np.cos(np.deg2rad(theta)))
        elif y_i >= max_y:
            y_i = max_y - 1
            x_i = int((rho - y_i * np.sin(np.deg2rad(theta))) / np.cos(np.deg2rad(theta)))

        x_f = max_x - 1
        y_f = int((rho - x_f * np.cos(np.deg2rad(theta))) / np.sin(np.deg2rad(theta)))

        if y_f < 0:
            y_f = 0
            x_f = int(rho / np.cos(np.deg2rad(theta)))
        elif y_f >= max_y:
            y_f = max_y - 1
            x_f = int((rho - y_f * np.sin(np.deg2rad(theta))) / np.cos(np.deg2rad(theta)))

        plt.plot([x_i, x_f], [y_i, y_f], color='red', linewidth=1)
    plt.show()
    fig.savefig(result_dir + 'hough.png')


def hough_findlines(image, hessian_matrix, result_dir):
    y, x = hessian_matrix.shape
    bin_x = 181
    bin_y = np.sqrt(x * x + y * y) * 2
    accumulator = np.zeros((int(bin_y), bin_x))

    points = detect_points(hessian_matrix)

    for point in points:
        for theta in range(bin_x):
            rho = point[1] * np.cos(np.deg2rad(theta)) + point[0] * np.sin(np.deg2rad(theta))
            accumulator[int(rho)][theta] += 1

    best_lines_polar = find_maxima(accumulator)
    plot_lines(image, (y, x), best_lines_polar, result_dir)

    return accumulator * 100

