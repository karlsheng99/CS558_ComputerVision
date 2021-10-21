import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def detect_points(source):
    row, col = source.shape
    points = []
    for i in range(row):
        for j in range(col):
            if source[i][j] != 0:
                points.append((i, j))

    return points


def find_line(point_1, point_2):
    x1, y1 = point_1
    x2, y2 = point_2
    if x1 == x2:
        m = np.inf
    else:
        m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    return m, b


def distance_point2line(slope, intercept, point):
    x2, y2 = point
    m1 = slope
    b1 = intercept
    if slope == 0:
        m2 = - np.inf
    else:
        m2 = - 1 / slope
    b2 = y2 - m2 * x2
    x1 = (b2 - b1) / (m1 - m2)
    y1 = m1 * x1 + b1

    dist = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    return dist


# def plot_pointsNline(source, points_subset):


def ransac(points, used_subset, min_inlier, dist_threshold):
    # Initial number of points
    s = 2
    # Distance threshold
    t = dist_threshold
    # Outlier ratio
    e = 0.5
    # Probability of a least one random sample is free from outliers
    p = 0.99
    # Number of samples
    n = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - e) ** s)))

    # i = 0  # iteration

    best_subset = []
    best_num_inlier = 0

    for i in range(n):
        inlier_subset = []

        index_1 = np.random.randint(len(points))
        while index_1 in used_subset:
            index_1 = np.random.randint(len(points))

        index_2 = np.random.randint(len(points))
        while index_2 in used_subset:
            index_2 = np.random.randint((len(points)))

        slope, intercept = find_line(points[index_1], points[index_2])

        num_inlier = 0

        for index in range(len(points)):
            if distance_point2line(slope, intercept, points[index]) <= dist_threshold:
                num_inlier += 1
                inlier_subset.append(index)

        if num_inlier > best_num_inlier:
            best_num_inlier = num_inlier
            best_subset.clear()
            best_subset = inlier_subset.copy()

        if num_inlier >= min_inlier:
            break

    return best_subset


def ransac_findlines(image, hessian_matrix, min_inlier, dist_threshold, result_dir):
    points = detect_points(hessian_matrix)
    used_subset = np.array([])
    lines = []

    i = 0
    while i < 4:
        find_subset = np.array(ransac(points, used_subset, min_inlier, dist_threshold))
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        left_most = points[find_subset[0]]
        right_most = points[find_subset[0]]

        for index in find_subset:
            y, x = points[index]
            rect = patches.Rectangle((x, y), 3, 3, facecolor='r')
            ax.add_patch(rect)

            if x < left_most[1]:
                left_most = points[index]
            elif x > right_most[1]:
                right_most = points[index]

        line = (left_most, right_most)
        plt.plot([left_most[1], right_most[1]], [left_most[0], right_most[0]], color='red', linewidth=1)
        plt.show()

        print('Is it a correct line? (y/n)')
        x = input()
        if x == 'y':
            used_subset = np.append(used_subset, find_subset)
            lines.append(line)
            i += 1

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for index in used_subset:
        y, x = points[int(index)]
        rect = patches.Rectangle((x, y), 3, 3, facecolor='r')
        ax.add_patch(rect)

    for line in lines:
        plt.plot([line[0][1], line[1][1]], [line[0][0], line[1][0]], color='red', linewidth=1)

    plt.show()

    fig.savefig(result_dir + 'ransac.png')

