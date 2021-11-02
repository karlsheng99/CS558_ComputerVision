from PIL import Image
import numpy as np
import convolution
import harris
import matching


def main():
    img1 = Image.open('AlignmentTwoViews/uttower_left.jpg').convert('L')
    img2 = Image.open('AlignmentTwoViews/uttower_right.jpg').convert('L')
    matrix1 = np.array(img1)
    matrix2 = np.array(img2)
    r1 = harris.harris_detector(matrix1, 1, 3000)
    r2 = harris.harris_detector(matrix2, 1, 3000)

    x = matching.ssd(img1, img2, matrix1, matrix2, r1, r2)
    print(x)



if __name__ == '__main__':
    main()
