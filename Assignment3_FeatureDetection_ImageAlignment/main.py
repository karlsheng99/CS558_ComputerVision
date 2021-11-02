from PIL import Image
import numpy as np
import convolution
import harris


def main():
    img = Image.open('AlignmentTwoViews/uttower_right.jpg').convert('L')
    matrix = np.array(img)
    r = harris.harris_detector(matrix, 1, 3000)
    rr = Image.fromarray(r)
    rr.show()



if __name__ == '__main__':
    main()
