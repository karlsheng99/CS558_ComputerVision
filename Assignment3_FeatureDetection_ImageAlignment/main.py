from PIL import Image, ImageOps
import numpy as np
import convolution
import harris


def main():
    img = Image.open('AlignmentTwoViews/uttower_right.jpg')
    matrix = np.array(ImageOps.grayscale(img))
    r = harris.harris_detector(matrix, 1, 3000)
    rr = Image.fromarray(r)
    rr.show()



if __name__ == '__main__':
    main()
