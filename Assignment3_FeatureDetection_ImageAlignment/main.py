from PIL import Image, ImageOps
import numpy as np
import convolution
import harris


def main():
    img = Image.open('AlignmentTwoViews/uttower_left.jpg')
    matrix = np.array(ImageOps.grayscale(img))
    r = harris.harris_detector(matrix, 1, 3000)
    rr = Image.fromarray(r)
    rr.show()
    print(r)



if __name__ == '__main__':
    main()
