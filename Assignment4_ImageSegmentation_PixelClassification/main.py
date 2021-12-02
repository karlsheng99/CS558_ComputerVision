from PIL import Image
from pathlib import Path
import numpy as np
import kmeans
import slic
import test


def main():
    image = Image.open('images/wt_slic.png')
    image_matrix = np.array(image)
    # clustered_matrix = kmeans.k_mean(image_matrix)
    # image_out = Image.fromarray(clustered_matrix.astype('uint8'), 'RGB')
    # image_out.show()
    clustered_matrix = slic.slic(image_matrix)
    image_out = Image.fromarray(clustered_matrix.astype('uint8'), 'RGB')
    image_out.show()
    #test.slic(image_matrix)


if __name__ == '__main__':
    main()
