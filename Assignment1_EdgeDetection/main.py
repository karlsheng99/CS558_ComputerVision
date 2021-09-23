import sys
import os.path
from os import path
import numpy as np
from PIL import Image
import gaussian
import convolution
import gradient
import suppression




def edge_detection():
    threshold = 80
    im = Image.open('images/kangaroo.pgm')
    im.show()
    # im.save('in.png')
    source = np.array(im)
    im.close()
    kernel = gaussian.gaussian_filter(1)
    smoothed_pix = convolution.convolve(source, kernel)
    gradient_magnitude, gradient_direction = gradient.compute_gradient(smoothed_pix, threshold)
    out_pix = suppression.non_max_suppress(gradient_magnitude, gradient_direction)
    out_pic = Image.fromarray(out_pix)
    out_pic.show()


def main():
    """ TODO:
        - argument
            - filename
                - default = kangaroo.pgm?
            - sigma
                - convolution*2 = sqrt(2)*sigma (sigma = 2^n preferred)
                - need for loop
                - default = 1
            - threshold
                - default = 80
        - save file
            - applied gaussian (smoothed image; name: sigma)
            - applied sobel filter (smoothed + edge detected image; name: sigma + threshold)
            - applied non-max suppression (suppressed image; name: sigma + threshold)
        -pledge
    """

    image_path = "images/red.pgm"
    sigma = 1
    threshold = 80

    if len(sys.argv) == 1:
        print("Run:", sys.argv[0], "file='images/red.pgm' sigma=1 threshold=80 by default")
    elif len(sys.argv) >= 2:
        file_path = "images/" + sys.argv[1]
        if not path.exists(file_path):
            print("Error: file does not exist")
            return 1
        else:
            image_path = file_path

        if len(sys.argv) == 2:
            print("Run:", sys.argv[0], sys.argv[1], "sigma=1 threshold=80 by default")


if __name__ == '__main__':
    main()
