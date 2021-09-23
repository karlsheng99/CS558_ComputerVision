import numpy as np
from PIL import Image
import gaussian
import convolution
import sobel
import suppression


def main():
    im = Image.open('images/red.pgm')
    im.show()
    # im.save('in.png')
    source = np.array(im)
    im.close()
    kernel = gaussian.gaussian_filter(1)
    smoothed_pix = convolution.convolve(source, kernel)
    # out_pic = Image.fromarray(out_pix)
    # out_pic.show()
    # out_pic.convert("L").save('myout5.png')
    #sobel.sobel_filter(smoothed_pix)
    #out_pix = sobel.get_gradient_magnitude(80)
    out_pix = suppression.suppression(smoothed_pix, 80)
    out_pic = Image.fromarray(out_pix)
    out_pic.show()


if __name__ == '__main__':
    main()
