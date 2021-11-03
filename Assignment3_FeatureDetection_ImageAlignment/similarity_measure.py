from PIL import Image
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


def plot_matching_features(source_image_1, source_image_2, best_20_pairs, window_size, title):
    new_img = Image.new('RGB', (source_image_1.width + source_image_2.width, source_image_1.height))
    new_img.paste(source_image_1, (0, 0))
    new_img.paste(source_image_2, (source_image_1.width, 0))

    fig, ax = plt.subplots(1)
    ax.imshow(new_img)

    for pair in best_20_pairs:
        y1, x1 = pair[0]
        y2, x2 = pair[1]

        rect_1 = patches.Rectangle((x1, y1), window_size, window_size, linewidth=1, edgecolor='r', facecolor='none')
        rect_2 = patches.Rectangle((x2 + source_image_1.width, y2), window_size, window_size, linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(rect_1)
        ax.add_patch(rect_2)

        plt.plot([x1, x2 + source_image_1.width], [y1, y2], color='red', linewidth=1)

    plt.title(title)
    plt.show()


def patch_similarity_matching(source_image_1, source_image_2, source_matrix_1, source_matrix_2, harris_matrix_1, harris_matrix_2, window_size=7):
    points_1 = detect_points(harris_matrix_1)
    points_2 = detect_points(harris_matrix_2)
    window_half_size = int(window_size / 2)

    # Keep tracking the coordinates of top 20 matching points
    best_20_ssd_pairs = []
    best_20_ncc_pairs = []

    # Keep tracking the ssd value of top 20 matching points
    best_20_ssd = np.full(20, np.inf)
    best_20_ncc = np.full(20, - np.inf)

    for point_1 in points_1:
        y1, x1 = point_1
        if y1 < window_half_size or x1 < window_half_size or \
                y1 > (len(source_matrix_1) - window_half_size - 1) or \
                x1 > (len(source_matrix_1[0]) - window_half_size - 1):
            break
        window_1 = np.array(source_matrix_1[y1 - window_half_size: y1 + window_half_size + 1,
                                            x1 - window_half_size: x1 + window_half_size + 1])
        for point_2 in points_2:
            y2, x2 = point_2
            if y2 < window_half_size or x2 < window_half_size or \
                    y2 > (len(source_matrix_2) - window_half_size - 1) or \
                    x2 > (len(source_matrix_2[0]) - window_half_size - 1):
                break
            window_2 = np.array(source_matrix_2[y2 - window_half_size: y2 + window_half_size + 1,
                                                x2 - window_half_size: x2 + window_half_size + 1])

            ssd_value = np.sum((window_1 - window_2) ** 2)

            mean_1 = np.mean(window_1)
            mean_2 = np.mean(window_2)

            numerator = np.sum((window_1 - mean_1) * (window_2 - mean_2))
            denominator = np.sqrt(np.sum((window_1 - mean_1) ** 2) * np.sum((window_2 - mean_2) ** 2))

            ncc_value = numerator / denominator

            for i in range(len(best_20_ssd)):
                if ssd_value < best_20_ssd[i]:
                    best_20_ssd = np.insert(best_20_ssd, i, ssd_value)
                    best_20_ssd = best_20_ssd[:-1]

                    best_20_ssd_pairs.insert(i, [point_1, point_2])
                    if len(best_20_ssd_pairs) > len(best_20_ssd):
                        best_20_ssd_pairs = best_20_ssd_pairs[:-1]

                    break

                if ncc_value > best_20_ncc[i]:
                    best_20_ncc = np.insert(best_20_ncc, i, ncc_value)
                    best_20_ncc = best_20_ncc[:-1]

                    best_20_ncc_pairs.insert(i, [point_1, point_2])
                    if len(best_20_ncc_pairs) > len(best_20_ncc):
                        best_20_ncc_pairs = best_20_ncc_pairs[:-1]

                    break

    plot_matching_features(source_image_1, source_image_2, best_20_ssd_pairs, window_size, 'SSD')
    plot_matching_features(source_image_1, source_image_2, best_20_ncc_pairs, window_size, 'NCC')

    return best_20_ssd_pairs, best_20_ncc_pairs
