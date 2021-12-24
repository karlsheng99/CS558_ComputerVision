import numpy as np


def separate_object(source_matrix, mask_matrix):
    row, col, rgb = source_matrix.shape
    object = []
    non_object = []

    for i in range(row):
        for j in range(col):
            pixel = mask_matrix[i][j]
            r = pixel[0]
            g = pixel[1]
            b = pixel[2]

            if r == 255:
                object.append((i, j))
            else:
                non_object.append((i, j))

    return object, non_object


def distance_pixel2center(pixel, center):
    pixel_r, pixel_g, pixel_b, pixel_x, pixel_y = pixel
    center_r, center_g, center_b, center_x, center_y = center

    distance_r = int(pixel_r) - int(center_r)
    distance_g = int(pixel_g) - int(center_g)
    distance_b = int(pixel_b) - int(center_b)
    distance_x = int(pixel_x) / 3 - int(center_x) / 3
    distance_y = int(pixel_y) / 3 - int(center_y) / 3
    distance = np.sqrt(distance_r ** 2 + distance_g ** 2 + distance_b ** 2 + distance_x ** 2 + distance_y ** 2)

    return distance


def find_cluster_mean(cluster_list, source_matrix):
    cluster_mean = []

    for cluster in cluster_list:
        size = len(cluster)
        sum_r = 0
        sum_g = 0
        sum_b = 0
        sum_x = 0
        sum_y = 0

        for pixel in cluster:
            y, x = pixel
            r = source_matrix[y][x][0]
            g = source_matrix[y][x][1]
            b = source_matrix[y][x][2]

            sum_r += int(r)
            sum_g += int(g)
            sum_b += int(b)
            sum_x += x
            sum_y += y

        mean_r = round(sum_r / size)
        mean_g = round(sum_g / size)
        mean_b = round(sum_b / size)
        mean_x = round(sum_x / size)
        mean_y = round(sum_y / size)

        cluster_mean.append((mean_r, mean_g, mean_b, mean_x, mean_y))

    return cluster_mean


def k_mean(train, pixel_set, num_cluster):
    convergence = False
    iteration = 1

    # find 10 random initial seeds
    cluster_center = []  # (r, g, b, x, y) of the cluster center pixel

    while len(cluster_center) < num_cluster:
        temp = np.random.randint(len(pixel_set))
        y, x = pixel_set[temp]
        r = train[y][x][0]
        g = train[y][x][1]
        b = train[y][x][2]
        color = (r, g, b, x, y)

        while color in cluster_center:
            temp = np.random.randint(len(pixel_set))
            y, x = pixel_set[temp]
            r = train[y][x][0]
            g = train[y][x][1]
            b = train[y][x][2]
            color = (r, g, b, x, y)

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
            pixel_color = (pixel_r, pixel_g, pixel_b, x, y)

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


def train(train_matrix, mask_matrix, num_cluster, train_object=True):
    # separate object and non_object sets
    object, non_object = separate_object(train_matrix, mask_matrix)

    target_vw = []
    # 10 color visual words for object and non_object sets
    if train_object:
        print('Object set')
        object_vw = k_mean(train_matrix, object, num_cluster)
        target_vw = object_vw
    else:
        print('Non-object set')
        non_object_vw = k_mean(train_matrix, non_object, num_cluster)
        target_vw = non_object_vw


    return target_vw


def test(test_matrix, object_vw, non_object_vw):
    row, col, rgb = test_matrix.shape
    background_matrix = test_matrix.copy()
    object_matrix = test_matrix.copy()

    count = 1
    for i in range(row):
        for j in range(col):
            r = test_matrix[i][j][0]
            g = test_matrix[i][j][1]
            b = test_matrix[i][j][2]
            pixel_color = (r, g, b, j, i)

            # find nearest word
            min_dist = np.inf
            is_object = False

            for vw in non_object_vw:
                dist = distance_pixel2center(pixel_color, vw)
                if dist < min_dist:
                    min_dist = dist

            for vw in object_vw:
                dist = distance_pixel2center(pixel_color, vw)
                if dist < min_dist:
                    is_object = True
                    min_dist = dist

            if is_object:
                background_matrix[i][j] = [255, 0, 0]
            else:
                object_matrix[i][j] = [0,0,0]

            x = int(count * 100 / (row * col))
            print('Generating test image: ' + '-' * x + '> ' + str(x) + '%')
            count += 1

    return background_matrix, object_matrix
