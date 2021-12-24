import numpy as np
from skimage import exposure


def compute_histogram(source_matrix):
    histogram, bin_center = exposure.histogram(source_matrix, nbins=8)

    return histogram

    
def distance_3d(p1, p2):
    dist = np.linalg.norm(p1 - p2)

    return dist


def find_mean(window):
    mean_vector = np.mean(window, axis=0)

    return mean_vector


def mean_shift(set, distance):
    cluster_center = []
    cluster_points = []

    for i in range(len(set)):
        point = set[i]
        converge = False
        center = point
        window = []
        for other_point in set:
            dist = distance_3d(center, other_point)
            if dist < distance:
                window.append(other_point)
        
        # find mean in the window
        while not converge:
            new_center = find_mean(window)
            if distance_3d(center, new_center) == 0:
                converge = True
                break

            center = new_center

            window.clear()
            for other_point in set:
                dist = distance_3d(center, other_point)
                if dist < distance:
                    window.append(other_point)

            

        # assign the point to the cluster center
        if center not in np.array(cluster_center):
            cluster_center.append(center)
            cluster_points.append([])

        #index = cluster_center.index(center)
        for index in range(len(cluster_center)):
            # temp_array = np.array(cluster_center)
            if np.all((center - cluster_center[index])) == 0:
                break
        cluster_points[index].append(i)
        print('point ' + str(i) + ' done!')

    return cluster_points

        