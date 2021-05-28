from skimage.transform import resize
from skimage import util
from PIL import Image
import numpy as np


class Params:
    def __init__(self):
        face_w = 0
        face_h = 0
        x1 = 0
        x2 = 0
        y1 = 0
        y2 = 0


def get_params(image, dets):
    p = Params()
    p.face_w = dets[0].right() - dets[0].left()
    p.face_h = dets[0].bottom() - dets[0].top()
    p.x1 = max(dets[0].left() - p.face_w // 2, 0)
    p.x2 = min(dets[0].right() + p.face_w // 2, image.shape[1])
    p.y1 = max(dets[0].top() - p.face_h // 2, 0)
    p.y2 = min(dets[0].bottom() + p.face_h // 2, image.shape[0])
    return p


def crop_face(image, detector, height, width):
    dets = detector(image, 1)
    if len(dets) != 1:
        return -1, -1, -1
    p = get_params(image, dets)
    cropped_image = image[p.y1:p.y2, p.x1:p.x2]
    # test cropped image/mask
    # imsave(crp_labels_path + filename + '.png', cropped_image)
    # cropped_hair_mask.save(crp_labels_path + filename + '_mask.png')
    # break
    resized_image = util.img_as_ubyte(resize(cropped_image, (height, width)))
    # test resized image/mask
    # imsave(crp_labels_path + filename + '.png', resized_image)
    return resized_image, dets, p


def resize_mask(hair_mask, height, width):
    hair_mask = Image.fromarray((hair_mask * 255).astype('uint8'), mode='L')
    resized_hair_mask = hair_mask.resize((width, height))
    pixels = np.asarray(resized_hair_mask.getdata())
    pixels = np.reshape(pixels, (height, width))
    np_hair_mask = np.zeros((pixels.shape[0], pixels.shape[1]), dtype='uint8')
    for i in range(pixels.shape[0]):
        for j in range(pixels.shape[1]):
            np_hair_mask[i, j] = pixels[i, j] > 0
    return np_hair_mask
