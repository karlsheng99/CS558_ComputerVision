import numpy as np
from PIL import Image
import gaussian
import convolution


def main():
    im = Image.open('images/kangaroo.pgm')
    im.show()
    im.save('in.png')
    source = np.array(im)
    im.close()
    kernel = gaussian.gaussian_filter(1)
    out_pix = convolution.convolve(source, kernel)
    out_pic = Image.fromarray(out_pix)
    out_pic.show()
    out_pic.convert("L").save('out.png')


if __name__ == '__main__':
    main()
