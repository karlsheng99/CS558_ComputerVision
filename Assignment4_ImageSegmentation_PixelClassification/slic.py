import numpy as np
import gradient


def distance_pixel2center(pixel, center):
    pixel_y, pixel_x, pixel_r, pixel_g, pixel_b = pixel
    center_y, center_x, center_r, center_g, center_b = center

    distance_x = int(pixel_x) / 2 - int(center_x) / 2
    distance_y = int(pixel_y) / 2 - int(center_y) / 2
    distance_r = int(pixel_r) - int(center_r)
    distance_g = int(pixel_g) - int(center_g)
    distance_b = int(pixel_b) - int(center_b)
    distance = np.sqrt(distance_x ** 2 + distance_y ** 2 + distance_r ** 2 + distance_g ** 2 + distance_b ** 2)

    return distance


def find_new_centroid(cluster_list):
    new_center = []
    new_center_color = []

    for i in range(len(cluster_list)):
        size = len(cluster_list[i])
        sum_x = 0
        sum_y = 0
        sum_r = 0
        sum_g = 0
        sum_b = 0

        for pixel in cluster_list[i]:
            y, x, r, g, b = pixel

            sum_x += int(x)
            sum_y += int(y)
            sum_r += int(r)
            sum_g += int(g)
            sum_b += int(b)

        mean_x = int(sum_x / size)
        mean_y = int(sum_y / size)
        mean_r = int(sum_r / size)
        mean_g = int(sum_g / size)
        mean_b = int(sum_b / size)

        new_center.append((mean_y, mean_x))
        new_center_color.append((mean_r, mean_g, mean_b))

    return new_center, new_center_color


def slic(source_matrix):
    row, col, rgb = source_matrix.shape
    clustered_matrix = np.empty((row, col, 3))
    clustered_matrix_border = clustered_matrix.copy()
    block_size = 50
    convergence = False
    iteration = 0
    max_iteration = 3

    # Initialization
    center = []
    center_color = []

    for center_y in range(int(block_size / 2), row, block_size):
        for center_x in range(int(block_size / 2), col, block_size):
            center.append((center_y, center_x))
            r = source_matrix[center_y][center_x][0]
            g = source_matrix[center_y][center_x][1]
            b = source_matrix[center_y][center_x][2]
            center_color.append((r, g, b))

    # Local Shift
    gradient_magnitude = gradient.compute_gradient(source_matrix)

    while not convergence and iteration < max_iteration:
        for c in range(len(center)):
            y, x = center[c]
            min_gradient = np.inf

            for i in range(y - 1, y + 2):
                for j in range(x - 1, x + 2):
                    temp_gradient = gradient_magnitude[i][j]

                    if temp_gradient < min_gradient:
                        min_gradient = temp_gradient
                        center[c] = (i, j)
                        r = source_matrix[i][j][0]
                        g = source_matrix[i][j][1]
                        b = source_matrix[i][j][2]
                        center_color[c] = (r, g, b)

        # Centroid Update
        cluster_list = []
        cluster_list_xy = []
        for i in range(len(center)):
            cluster_list.append([])
            cluster_list_xy.append([])

        count = 1
        for i in range(row):
            for j in range(col):
                pixel_r = source_matrix[i][j][0]
                pixel_g = source_matrix[i][j][1]
                pixel_b = source_matrix[i][j][2]
                p = (i, j, pixel_r, pixel_g, pixel_b)
                min_dist = np.inf
                centroid_i = 0

                for k in range(len(center)):
                    center_y, center_x = center[k]
                    center_r, center_g, center_b = center_color[k]
                    dist_xy = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
                    if dist_xy < 100:
                        c = (center_y, center_x, center_r, center_g, center_b)
                        dist = distance_pixel2center(p, c)

                        if dist < min_dist:
                            min_dist = dist
                            centroid_i = k

                cluster_list[centroid_i].append(p)
                cluster_list_xy[centroid_i].append((i, j))

                x = int(count * 100 / (row * col))
                print('Iteration ' + str(iteration + 1) + ': ' + '-' * x + '> ' + str(x) + '%')
                count += 1
        new_center, new_center_color = find_new_centroid(cluster_list)

        if new_center == center and new_center_color == center_color:
            convergence = True

        center = new_center
        center_color = new_center_color
        iteration += 1

        if convergence or iteration == max_iteration:
            for i in range(len(cluster_list)):
                new_r, new_g, new_b = new_center_color[i]
                for pixel in cluster_list[i]:
                    y, x, r, g, b = pixel
                    clustered_matrix[y][x][0] = new_r
                    clustered_matrix[y][x][1] = new_g
                    clustered_matrix[y][x][2] = new_b

            # Add border
            clustered_matrix_border = clustered_matrix.copy()

            count = 1
            drawn_pixel = np.zeros((row, col))
            for i in range(len(cluster_list_xy)):
                for pixel in cluster_list_xy[i]:
                    y, x = pixel
                    is_border = False

                    if 0 < x < col - 1 and 0 < y < row - 1:
                        for p in range(y - 1, y + 2):
                            for q in range(x - 1, x + 2):
                                r, g, b = clustered_matrix_border[p][q]
                                if (p, q) not in cluster_list_xy[i]:
                                    is_border = True
                                    drawn_pixel[p][q] = 1

                        if is_border and drawn_pixel[y][x] == 0:
                            clustered_matrix_border[y][x] = [0, 0, 0]

                    x = int(count * 100 / (row * col))
                    print('Generating clustered image with border: ' + '-' * x + '> ' + str(x) + '%')
                    count += 1

    return clustered_matrix, clustered_matrix_border

