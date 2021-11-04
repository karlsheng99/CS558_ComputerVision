from PIL import Image
from pathlib import Path
import numpy as np
import harris
import similarity_measure
import image_alignment


def main():
    img_1 = Image.open('AlignmentTwoViews/uttower_left.jpg')
    img_2 = Image.open('AlignmentTwoViews/uttower_right.jpg')
    result_dir = 'results/'
    if not Path(result_dir).exists():
        Path(result_dir).mkdir()

    source_matrix_1 = np.array(img_1.convert('L'))
    source_matrix_2 = np.array(img_2.convert('L'))

    top_1000_features_1, harris_matrix_1 = harris.harris_detector(source_matrix_1, 1)
    top_1000_features_2, harris_matrix_2 = harris.harris_detector(source_matrix_2, 1)

    top_1000_img_1 = Image.fromarray(top_1000_features_1)
    top_1000_img_1.convert('L').save(result_dir + '/left_top_1000_features.png')
    top_1000_img_2 = Image.fromarray(top_1000_features_2)
    top_1000_img_2.convert('L').save(result_dir + '/right_top_1000_features.png')

    harris_img_1 = Image.fromarray(harris_matrix_1)
    harris_img_1.convert('L').save(result_dir + '/left_non_max_suppression.png')
    harris_img_2 = Image.fromarray(harris_matrix_2)
    harris_img_2.convert('L').save(result_dir + '/right_non_max_suppression.png')

    ssd, ncc = similarity_measure.patch_similarity_matching(img_1, img_2, source_matrix_1, source_matrix_2,
                                                            harris_matrix_1, harris_matrix_2, result_dir, 15)

    img_r = img_1.rotate(45, expand=True)
    source_matrix_r = np.array(img_r.convert('L'))
    top_1000_features_r, harris_matrix_r = harris.harris_detector(source_matrix_r, 1)
    ssd_r, ncc_r = similarity_measure.patch_similarity_matching(img_r, img_2, source_matrix_r, source_matrix_2,
                                                            harris_matrix_r, harris_matrix_2, result_dir, 15, '_rotated')

    affine_vector = image_alignment.affine_transform(harris_matrix_1, harris_matrix_2, ncc)
    merged_matrix = image_alignment.panorama_stitching(source_matrix_1, source_matrix_2, affine_vector)
    merged_img = Image.fromarray(merged_matrix)
    merged_img.convert('L').save(result_dir + '/panorama_stitching.png')


if __name__ == '__main__':
    main()
