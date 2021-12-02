import numpy as np


def distance_pixel2center(pixel, center):
    pixel_r, pixel_g, pixel_b = pixel
    center_r, center_g, center_b = center

    distance_r = int(pixel_r) - int(center_r)
    distance_g = int(pixel_g) - int(center_g)
    distance_b = int(pixel_b) - int(center_b)
    distance = np.sqrt(distance_r ** 2 + distance_g ** 2 + distance_b ** 2)

    return distance


def find_cluster_mean(cluster_list, source_matrix):
    cluster_mean = []

    for i in range(len(cluster_list)):
        cluster_i = cluster_list[i]
        size = len(cluster_i)
        sum_r = 0
        sum_g = 0
        sum_b = 0

        for pixel in cluster_i:
            r, g, b = source_matrix[pixel[0]][pixel[1]]
            sum_r += int(r)
            sum_g += int(g)
            sum_b += int(b)

        mean_r = int(sum_r / size)
        mean_g = int(sum_g / size)
        mean_b = int(sum_b / size)

        cluster_mean.append((mean_r, mean_g, mean_b))

    return cluster_mean


def k_mean(source_matrix):
    row, col, rgb = source_matrix.shape
    clustered_matrix = np.empty((row, col, 3))
    convergence = False
    iteration = 1

    # find 10 random pixels as initial seeds
    cluster_center = []  # (r, g, b) color of the cluster center pixel

    while len(cluster_center) < 10:
        x = np.random.randint(col)
        y = np.random.randint(row)

        r = source_matrix[y][x][0]
        g = source_matrix[y][x][1]
        b = source_matrix[y][x][2]
        color = (r, g, b)

        while color in cluster_center:
            x = np.random.randint(col)
            y = np.random.randint(row)

            r = source_matrix[y][x][0]
            g = source_matrix[y][x][1]
            b = source_matrix[y][x][2]
            color = (r, g, b)

        cluster_center.append(color)

    while not convergence:
        print('Iteration ' + str(iteration) + ':', cluster_center)
        cluster_list = []
        for i in range(len(cluster_center)):
            cluster_list.append([])

        for i in range(row):
            for j in range(col):
                pixel_r = source_matrix[i][j][0]
                pixel_g = source_matrix[i][j][1]
                pixel_b = source_matrix[i][j][2]
                pixel_color = (pixel_r, pixel_g, pixel_b)
                pixel = (i, j)
                min_dist = np.inf
                cluster_i = 0

                for k in range(len(cluster_center)):
                    distance = distance_pixel2center(pixel_color, cluster_center[k])

                    if distance < min_dist:
                        min_dist = distance
                        cluster_i = k

                cluster_list[cluster_i].append(pixel)

        # find cluster mean
        cluster_mean = find_cluster_mean(cluster_list, source_matrix)

        if cluster_mean == cluster_center:
            convergence = True

            for i in range(len(cluster_list)):
                r, g, b = cluster_mean[i]
                for pixel in cluster_list[i]:
                    y, x = pixel
                    clustered_matrix[y][x][0] = r
                    clustered_matrix[y][x][1] = g
                    clustered_matrix[y][x][2] = b

        cluster_center = cluster_mean
        iteration += 1

    return clustered_matrix

