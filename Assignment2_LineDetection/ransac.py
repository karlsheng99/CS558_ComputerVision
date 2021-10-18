import numpy as np

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

    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    return m, b


def distance_point2line(slope, intercept, point):
    x2, y2 = point
    m1 = slope
    b1 = intercept
    m2 = - 1 / slope
    b2 = y2 - m2 * x2
    x1 = (b2 - b1) / (m1 - m2)
    y1 = m1 * x1 + b1

    dist = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    return dist


# def plot_pointsNline(source, points_subset):


def ransac(source, min_inlier, dist_threshold):
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
    points = detect_points(source)
    used_subset = []
    best_subset = []
    best_num_inlier = 0

    for i in range(n):
        inlier_subset = []

        index_1 = np.random.randint(len(points))
        while index_1 in used_subset:
            index_1 = np.random.randint(len(points))
        used_subset.append(index_1)
        inlier_subset.append(points[index_1])

        index_2 = np.random.randint(len(points))
        while index_2 in used_subset:
            index_2 = np.random.randint((len(points)))
        used_subset.append(index_2)
        inlier_subset.append(points[index_2])

        slope, intercept = find_line(points[index_1], points[index_2])

        num_inlier = 2

        for index in range(len(points)):
            if distance_point2line(slope, intercept, points[index]) <= dist_threshold:
                num_inlier += 1
                inlier_subset.append(points[index])

        if num_inlier > best_num_inlier:
            best_num_inlier = num_inlier
            best_subset.clear()
            best_subset = inlier_subset.copy()

        if num_inlier >= min_inlier:
            break

    return best_subset