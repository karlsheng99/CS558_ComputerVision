import numpy as np


def separate_sky(source_matrix, mask_matrix):
    row, col, rgb = source_matrix.shape
    sky = []
    non_sky = []

    for i in range(row):
        for j in range(col):
            pixel = mask_matrix[i][j]
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]

            if r == g == b == 255:
                sky.append((i, j))
            else:
                non_sky.append((i, j))

    return sky, non_sky


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

    for cluster in cluster_list:
        size = len(cluster)
        sum_r = 0
        sum_g = 0
        sum_b = 0

        for pixel in cluster:
            y, x = pixel
            r = source_matrix[y][x][0]
            g = source_matrix[y][x][1]
            b = source_matrix[y][x][2]

            sum_r += int(r)
            sum_g += int(g)
            sum_b += int(b)

        mean_r = int(sum_r / size)
        mean_g = int(sum_g / size)
        mean_b = int(sum_b / size)

        cluster_mean.append((mean_r, mean_g, mean_b))

    return cluster_mean


def k_mean(train, pixel_set):
    convergence = False
    iteration = 1

    # find 10 random initial seeds
    cluster_center = []  # (r, g, b) color of the cluster center pixel

    while len(cluster_center) < 10:
        temp = np.random.randint(len(pixel_set))
        y, x = pixel_set[temp]
        r = train[y][x][0]
        g = train[y][x][1]
        b = train[y][x][2]
        color = (r, g, b)

        while color in cluster_center:
            temp = np.random.randint(len(pixel_set))
            y, x = pixel_set[temp]
            r = train[y][x][0]
            g = train[y][x][1]
            b = train[y][x][2]
            color = (r, g, b)

        cluster_center.append(color)

    while not convergence:
        print('Iteration ' + str(iteration) + ':', cluster_center)

        cluster_list = []
        for i in range(len(cluster_center)):
            cluster_list.append([])

        # assign each pixel to the closest cluster center
        for pixel in pixel_set:
            y, x = pixel
            pixel_r = train[y][x][0]
            pixel_g = train[y][x][1]
            pixel_b = train[y][x][2]
            pixel_color = (pixel_r, pixel_g, pixel_b)

            min_dist = np.inf
            cluster_i = 0

            for i in range(len(cluster_center)):
                dist = distance_pixel2center(pixel_color, cluster_center[i])

                if dist < min_dist:
                    min_dist = dist
                    cluster_i = i

            cluster_list[cluster_i].append(pixel)

        # find cluster mean
        cluster_mean = find_cluster_mean(cluster_list, train)

        if cluster_mean == cluster_center:
            convergence = True

        cluster_center = cluster_mean
        iteration += 1

    return cluster_center


def train(train_matrix, mask_matrix):
    # separate sky and non_sky sets
    sky, non_sky = separate_sky(train_matrix, mask_matrix)

    # 10 color visual words for sky and non_sky sets
    print('Sky set')
    sky_vw = k_mean(train_matrix, sky)
    print('Non-sky set')
    non_sky_vw = k_mean(train_matrix, non_sky)

    return sky_vw, non_sky_vw


def test(test_matrix, sky_vw, non_sky_vw):
    row, col, rgb = test_matrix.shape
    classified_matrix = test_matrix.copy()

    count = 1
    for i in range(row):
        for j in range(col):
            r = test_matrix[i][j][0]
            g = test_matrix[i][j][1]
            b = test_matrix[i][j][2]
            pixel_color = (r, g, b)

            # find nearest word
            min_dist = np.inf
            is_sky = False

            for vw in non_sky_vw:
                dist = distance_pixel2center(pixel_color, vw)
                if dist < min_dist:
                    min_dist = dist

            for vw in sky_vw:
                dist = distance_pixel2center(pixel_color, vw)
                if dist < min_dist:
                    is_sky = True
                    min_dist = dist

            if is_sky:
                classified_matrix[i][j] = [255, 0, 0]

            x = int(count * 100 / (row * col))
            print('Generating test image: ' + '-' * x + '> ' + str(x) + '%')
            count += 1

    return classified_matrix

