from PIL import Image, ImageOps
from pathlib import Path
from skimage import img_as_float
import numpy as np
import ImageDataAssociation as ida
import segmentation as seg

def down_sample(source_image, ratio):
    x, y = source_image.size
    size = (int(x * ratio), int(y * ratio))
    out = source_image.resize(size)
    return out


def p1():
    set = []
    image_set = []
    input_path = Path('P1_set2/')
    for image_path in input_path.iterdir():
        source_image = Image.open(image_path)
        source_matrix = np.array(img_as_float(source_image))
        r_mat = source_matrix[:,:,0]
        g_mat = source_matrix[:,:,1]
        b_mat = source_matrix[:,:,2]

        # compute hog for each image
        hist_r = ida.compute_histogram(r_mat)
        hist_g = ida.compute_histogram(g_mat)
        hist_b = ida.compute_histogram(b_mat)
        hist_out = np.concatenate((hist_r, hist_g, hist_b))
        set.append(hist_out)
        image_set.append(source_image)

    print('compute histogram done!')
    
    # mean-shift
    cluster_points = ida.mean_shift(set, 1889900)

    output_path = 'results/P1/Set2/'
    for i in range(len(cluster_points)):
        path = output_path + str(i) + '/'
        if not Path(path).exists():
            Path(path).mkdir()
        for index in cluster_points[i]:
            image_set[index].save(path + str(index) + '.jpg')


def p2_train(mask_img, train_img, num_cluster, ratio=1, train_object=True):
    mask_mat = np.array(down_sample(mask_img, ratio))
    train_mat = np.array(down_sample(train_img, ratio))
    object_vw = seg.train(train_mat, mask_mat, num_cluster, train_object)

    return object_vw


def p2():
    # train
    mask1_img = Image.open('P2_set/train/wheel2.jpg')
    mask2_img = Image.open('P2_set/train/tractor_top.jpg')
    mask3_img = Image.open('P2_set/train/train.jpg')
    train_img = Image.open('P2_set/train/0000.jpg')
    wheel_vw = p2_train(mask1_img, train_img, 2, ratio=0.2)
    tractor_top_vw = p2_train(mask2_img, train_img, 8, ratio=0.2)
    background_vw = p2_train(mask3_img, train_img, 15, ratio=0.2, train_object=False)
    tractor_vw = wheel_vw + tractor_top_vw

    # test
    input_path = Path('P2_set/test')
    num = 0
    for image_path in input_path.iterdir():
        test_img = Image.open(image_path)
        test_mat = np.array(down_sample(test_img, 0.2))
        back_mat, obj_mat = seg.test(test_mat, tractor_vw, background_vw)
        back_img = Image.fromarray(back_mat.astype('uint8'), 'RGB')
        back_img.save('results/P2/back_' + str(num) + '.jpg')
        obj_img = Image.fromarray(obj_mat.astype('uint8'), 'RGB')
        obj_img.save('results/P2/obj_' + str(num) + '.jpg')
        num += 1

def main():
    # p1()
    p2()
    


if __name__ == '__main__':
    main()