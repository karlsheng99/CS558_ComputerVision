import sys
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import convolution
import gaussian
import hessian
import ransac
import hough

def main():
    # image = Image.open('road.png')
    # image.show()
    # image_pix = np.array(image)
    # image.close()
    #
    # gauss_filter = gaussian.gaussian_filter(1)
    # out_pix = convolution.convolve(image_pix, gauss_filter)
    #
    # #out = Image.fromarray(out_pix)
    # #out.show()
    #
    # hes_pix = hessian.hessian(out_pix)
    # hes = Image.fromarray(hes_pix)
    # hes.convert("L").save('hessian2.png')

    image = Image.open('hessian2.png')
    image2 = Image.open('road.png')
    image_pix = np.array(image)

    # ransac.ransac_findlines(image2, image_pix, 100, 2)
    out_pix = hough.hough_transform(image2, image_pix)
    out = Image.fromarray(out_pix)
    out.convert("L").save('hough_space.png')


if __name__ == '__main__':
    main()
