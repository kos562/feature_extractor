import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from skimage.io import imread


class MedianColor:
    def __init__(self):
        pass

    def calc_median_color_from_image(self, image, mask):
        """
            Calculate median RGB-color for mask-selected
            pixels from image.
              image: rgb image shape (height, width, 3)
              mask:  boolean nd.array (height, width, 1)
            return: r,b,g
        """
        r, g, b = cv2.split(image)
        return np.median(r[mask == 1]), np.median(g[mask == 1]), np.median(b[mask == 1])

    def trim_array_between_median(self, arr, factor=0.15):
        """
            Crop the histogram of one channel, pad to both sides of the median.
            factor:     float from 0 to 1, needed to
                        express the padding from the median in percent
        """
        colors_count = 256

        median = np.median(arr)

        left_border = max(median - colors_count * factor, 0)
        right_border = min(median + colors_count * factor, colors_count)

        # sorry for shitty way to make array
        cropped_arr = []
        for elem in arr:
            if left_border < elem < right_border:
                cropped_arr.append(elem)

        # cropped_arr = arr[arr > left_border]
        # cropped_arr = [cropped_arr < right_border]
        return np.array(cropped_arr)

    def calc_median_color_with_trim(self, image, mask, factor=0.15):
        r, g, b = cv2.split(image)

        r = np.median(self.trim_array_between_median(r[mask == 1], factor))
        g = np.median(self.trim_array_between_median(g[mask == 1], factor))
        b = np.median(self.trim_array_between_median(b[mask == 1], factor))

        return r, g, b

    def plot_and_save_hair_and_detected_color(self, image, mask, savepath):
        """
        For testing and visualize how hair color calculation works
        """
        image_ = image.copy()
        mask_ = mask.copy()

        r, g, b = self.calc_median_color_from_image(image_, mask_)

        fig, axs = plt.subplots(1, 5, figsize=(12, 3))

        # Remove ticks
        for i in range(len(axs)):
            axs[i].set_xticks([])
            axs[i].set_yticks([])

        axs[0].title.set_text("Original image")
        axs[1].title.set_text("Only hair")
        axs[2].title.set_text("Median")
        axs[2].set_xlabel("{},{},{}".format(int(r), int(g), int(b)))

        # Plot original image
        axs[0].imshow(image_)

        # Plot extracted hair
        image_[mask_ == 0] = [255, 255, 255]
        axs[1].imshow(image_)

        # Plot color
        image_[:, :, :] = [r, g, b]
        axs[2].imshow(image_)

        # #######################  TRY REG  ###################################
        image_ = image.copy()
        r, g, b = self.calc_median_color_with_trim(image_, mask_, 0.1)
        image_[:, :, :] = [r, g, b]
        axs[3].imshow(image_)
        axs[3].title.set_text("Median & {} reg.".format(0.05))
        axs[3].set_xlabel("{},{},{}".format(int(r), int(g), int(b)))
        ####################################################################
        image_ = image.copy()
        r, g, b = self.calc_median_color_with_trim(image_, mask_, 0.35)
        image_[:, :, :] = [r, g, b]
        axs[4].imshow(image_)
        axs[4].title.set_text("Median & {} reg.".format(0.35))
        axs[4].set_xlabel("{},{},{}".format(int(r), int(g), int(b)))
        #########################################################################

        plt.rcParams['savefig.facecolor'] = 'white'
        plt.savefig(savepath, dpi=450)

        return fig


def process():
    imgs_path = './data/test_photos/'
    labels_path = './data/annotations/cropped-hair-masks_iso_iec/'

    res_images_path = './data/hair-color/'

    if not os.path.exists(res_images_path):
        os.mkdir(res_images_path)

    dirs = os.walk(imgs_path)
    for path_from_top, subdirs, files in dirs:
        for f in files:
            filename = str(f).split('.')[0]
            try:
                hair_mask = np.load(labels_path + filename + '.npy')
            except:
                continue

            image_path = str(path_from_top) + '/' + str(f)
            image = imread(image_path)

            mc = MedianColor()
            mc.plot_and_save_hair_and_detected_color(image, hair_mask, res_images_path + filename)


if __name__ == "__main__":
    process()
