from PIL import Image
from pathlib import Path
import numpy as np
import kmeans
import slic
import classification


def kmeans_segmentation():
    source_image = Image.open('images/white-tower.png')
    source_matrix = np.array(source_image)
    clustered_matrix = kmeans.k_mean(source_matrix)
    clustered_image = Image.fromarray(clustered_matrix.astype('uint8'), 'RGB')
    clustered_image.save('results/k-means_segmentation.png')


def SLIC():
    source_image_1 = Image.open('images/white-tower.png')
    source_image_2 = Image.open('images/wt_slic.png')
    source_matrix_1 = np.array(source_image_1)
    source_matrix_2 = np.array(source_image_2)
    clustered_mat_1, clustered_mat_b_1 = slic.slic(source_matrix_1)
    clustered_img_1 = Image.fromarray(clustered_mat_1.astype('uint8'), 'RGB')
    clustered_img_1.save('results/slic_no_border_1.png')
    clustered_img_b_1 = Image.fromarray(clustered_mat_b_1.astype('uint8'), 'RGB')
    clustered_img_b_1.save('results/slic_with_border_1.png')
    clustered_mat_2, clustered_mat_b_2 = slic.slic(source_matrix_2)
    clustered_img_2 = Image.fromarray(clustered_mat_2.astype('uint8'), 'RGB')
    clustered_img_2.save('results/slic_no_border_2.png')
    clustered_img_b_2 = Image.fromarray(clustered_mat_b_2.astype('uint8'), 'RGB')
    clustered_img_b_2.save('results/slic_with_border_2.png')


def pixel_classification():
    train_image = Image.open('images/sky/sky_train.jpg')
    train_matrix = np.array(train_image)
    mask_image = Image.open('images/sky/non_sky_train.jpg')
    mask_matrix = np.array(mask_image)
    sky_visual_words, non_sky_visual_words = classification.train(train_matrix, mask_matrix)

    for i in range(1, 5):
        test_image = Image.open('images/sky/sky_test' + str(i) + '.jpg')
        test_matrix = np.array(test_image)
        classified_matrix = classification.test(test_matrix, sky_visual_words, non_sky_visual_words)
        classified_image = Image.fromarray(classified_matrix.astype('uint8'), 'RGB')
        classified_image.save('results/pixel_classification_' + str(i) + '.png')


def main():
    result_dir = 'results/'
    if not Path(result_dir).exists():
        Path(result_dir).mkdir()

    kmeans_segmentation()
    SLIC()
    pixel_classification()


if __name__ == '__main__':
    main()

