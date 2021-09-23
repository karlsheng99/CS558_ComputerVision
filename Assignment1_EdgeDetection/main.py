import sys
from pathlib import Path
from PIL import Image
import numpy as np
import gaussian
import convolution
import gradient
import suppression


def edge_detection(image, sigma, threshold):
    source_path = 'images/' + image
    source_pic = Image.open(source_path)
    result_dir = 'results/'
    if not Path(result_dir).exists():
        Path(result_dir).mkdir()
    image_name = image[:-4]
    source_pic.save(result_dir + image_name + '.png')

    source_pix = np.array(source_pic)
    source_pic.close()

    """
    sigma = 1 -> sigma=1 * 1
    sigma = 2 -> sigma=1 * 3
    sigma = 4 -> sigma=1 * 5
    sigma = 8 -> sigma=1 * 7
    """
    smoothed_pix = np.copy(source_pix)
    if sigma == 1 or sigma == 2 or sigma == 4 or sigma == 8:
        gauss_kernel = gaussian.gaussian_filter(1)
        for i in range(int(np.log2(sigma)) * 2 + 1):
            smoothed_pix = convolution.convolve(smoothed_pix, gauss_kernel)
    else:
        gauss_kernel = gaussian.gaussian_filter(sigma)
        smoothed_pix = convolution.convolve(smoothed_pix, gauss_kernel)

    smoothed_pic = Image.fromarray(smoothed_pix)
    smoothed_pic.convert("L").save(result_dir + image_name + '_gaussian_sigma=' + str(sigma) + '.png')

    gradient_magnitude, gradient_direction = gradient.compute_gradient(smoothed_pix, threshold)
    gradient_pic = Image.fromarray(gradient_magnitude)
    gradient_pic.convert("L").save(result_dir + image_name + '_gradient_sigma=' + str(sigma) + '_threshold=' + str(threshold) + '.png')

    suppression_pix = suppression.non_max_suppress(gradient_magnitude, gradient_direction)
    suppression_pic = Image.fromarray(suppression_pix)
    suppression_pic.convert("L").save(result_dir + image_name + '_suppression_sigma=' + str(sigma) + '_threshold=' + str(threshold) + '.png')


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

    filename = 'red.pgm'
    sigma = 4
    threshold = 80

    edge_detection(filename, sigma, threshold)

    # if len(sys.argv) == 1:
    #     print("Run:", sys.argv[0], "file='images/red.pgm' sigma=1 threshold=80 by default")
    # elif len(sys.argv) >= 2:
    #     file_path = "images/" + sys.argv[1]
    #     if not Path(file_path).exists():
    #         print("Error: file does not exist")
    #         return 1
    #     else:
    #         filename = sys.argv[1]
    #
    #     if len(sys.argv) == 2:
    #         print("Run:", sys.argv[0], sys.argv[1], "sigma=1 threshold=80 by default")


if __name__ == '__main__':
    main()
