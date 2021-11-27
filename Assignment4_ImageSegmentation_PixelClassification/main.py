from PIL import Image
from pathlib import Path
import numpy as np
import kmeans


def main():
    image = Image.open('images/white-tower.png')
    image_matrix = np.array(image)
    clustered_matrix = kmeans.k_mean(image_matrix)
    image_out = Image.fromarray(clustered_matrix.astype('uint8'), 'RGB')
    image_out.show()


if __name__ == '__main__':
    main()
