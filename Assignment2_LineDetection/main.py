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

    subset = ransac.ransac(image_pix, 20, 2)

    fig, ax = plt.subplots(1)
    ax.imshow(image2)

    p_top = subset[0]
    p_bot = subset[0]
    top_most = p_top[0]
    bot_most = p_bot[0]

    for i in subset:
        y, x = i
        rect = patches.Rectangle((x, y), 3, 3, facecolor='r')
        ax.add_patch(rect)

        if y < top_most:
            top_most = y
            p_top = i
        elif y > bot_most:
            bot_most = y
            p_bot = i

    plt.plot([p_top[1], p_bot[1]], [p_top[0], p_bot[0]], color='red', linewidth=1)
    plt.show()



if __name__ == '__main__':
    main()
