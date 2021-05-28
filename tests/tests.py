import unittest
import matplotlib.pyplot as plt
import numpy as np
from skimage import util
from skimage.transform import rescale

import src.utils.graphic_calcs as ugc
from src.get_colortypes_csv import \
    get_landmarks, \
    get_face_undereyes_color, \
    get_face_forehead_color, \
    get_face_cheeks_color, get_face_hair_color

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class Case1(unittest.TestCase):
    def setUp(self):
        self.image_path = './src/data/photo_augmentation/ColortypesDone2CropedAngelEurop/summer/00000428.jpg'
        self.image = plt.imread(self.image_path)
        self.image = util.img_as_ubyte(rescale(self.image, (0.5, 0.5, 1), anti_aliasing=False))
        self.imageBGR = ugc.switch_rgb(self.image)
        self.landmarks, _ = get_landmarks(self.image, '00000428.jpg')

        _, _, _, self.imageRGB = get_face_hair_color(self.imageBGR)

    def test_get_face_undereyes_color(self):
        face_undereyes_color_h, face_undereyes_color_s, \
            face_undereyes_color_v = get_face_undereyes_color(self.imageBGR, self.landmarks[1],
                                                              self.landmarks[29],
                                                              self.landmarks[15])
        self.assertEqual([face_undereyes_color_h, face_undereyes_color_s, face_undereyes_color_v],
                         [0.043478260869565154, 0.2090909090909091, 0.8627450980392157])

    def test_get_face_forehead_color(self):
        face_forehead_color_h, face_forehead_color_s, \
            face_forehead_color_v = get_face_forehead_color(self.imageBGR, self.landmarks)
        self.assertEqual([face_forehead_color_h, face_forehead_color_s, face_forehead_color_v],
                         [0.058641975308642014, 0.23175965665236056, 0.9137254901960784])

    def test_get_face_cheeks_color(self):
        face_cheeks_color_h, face_cheeks_color_s, \
            face_cheeks_color_v = get_face_cheeks_color(self.imageBGR, self.landmarks[1],
                                                       self.landmarks[15],
                                                       self.landmarks[12],
                                                       self.landmarks[4])
        self.assertEqual([face_cheeks_color_h, face_cheeks_color_s, face_cheeks_color_v],
                         [0.050847457627118696, 0.25764192139737996, 0.8980392156862745])

    def test_get_face_hair_color(self):
        face_hair_color_h, face_hair_color_s, \
            face_hair_color_v, _ = get_face_hair_color(self.imageBGR)
        self.assertEqual([face_hair_color_h, face_hair_color_s, face_hair_color_v],
                         [0.0, 0.0, 1.0])
