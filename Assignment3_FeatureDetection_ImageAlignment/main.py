from PIL import Image
import numpy as np
import convolution
import harris
import similarity_measure
import image_alignment


def main():
    img1 = Image.open('AlignmentTwoViews/uttower_left.jpg')
    img2 = Image.open('AlignmentTwoViews/uttower_right.jpg')
    # img1 = img1.rotate(45, expand=True)
    matrix1 = np.array(img1.convert('L'))
    matrix2 = np.array(img2.convert('L'))
    r1 = harris.harris_detector(matrix1, 1)
    r2 = harris.harris_detector(matrix2, 1)
    # rr1 = Image.fromarray(r1)
    # rr1.show()
    # rr2 = Image.fromarray(r2)
    # rr2.convert('L').save('non-max2.png')

    ssd, ncc = similarity_measure.patch_similarity_matching(img1, img2, matrix1, matrix2, r1, r2, 15)

    mat = image_alignment.affine_transform(r1, r2, ncc)

    imgo = img2.transform((r2.shape[1], r2.shape[0]), Image.AFFINE, (mat[0], mat[1], mat[2], mat[3], mat[4], mat[5]))
    imgo.show()


if __name__ == '__main__':
    main()
