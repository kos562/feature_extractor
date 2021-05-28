"""get colortypes"""


import os
import cv2
import dlib
import numpy as np
import pandas as pd
from imutils import face_utils
from skimage import util
from skimage.io import imread
from skimage.transform import rescale
from tqdm import tqdm
import src.moduleUnet as moduleUnet
import src.utils.graphic_calcs as ugc
import src.utils.graphic_transforms as ugt

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./src/data/weights/shape_predictor_68_face_landmarks.dat')

image_folder = './src/data/photo_augmentation/'
save_folder = './src/data/result/'

dirs = os.walk(image_folder)
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

pathToWeights = './src/data/weights/100.dat'
net = moduleUnet.SegmentationUnet()
net.loadModel(pathToWeights)

output_list = []
empty_faces = []


def get_landmarks(image_, f_):
    gray = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    continue_iter = False
    if len(rects) != 1:
        empty_faces.append(str(f_))
        continue_iter = True
    landmarks_ = predictor(gray, rects[0])
    landmarks_ = face_utils.shape_to_np(landmarks_)

    return landmarks_, continue_iter


def get_face_undereyes_color(image_bgr, landmarks_1, landmarks_29, landmarks_15):
    face_undereyes_color = ugc.get_color_of_line(image_bgr, landmarks_1, landmarks_29,
                                                 landmarks_15)

    return face_undereyes_color[0], face_undereyes_color[1], face_undereyes_color[2]


def get_face_forehead_color(image_bgr, landmarks_):
    face_forehead_color = ugc.get_forehead_color(image_bgr, landmarks_)

    return face_forehead_color[0], face_forehead_color[1], face_forehead_color[2]


def get_face_cheeks_color(image_bgr, landmarks_1, landmarks_15, landmarks_12, landmarks_4):
    face_cheeks_color = ugc.get_color_of_zone(image_bgr, [landmarks_1, landmarks_15,
                                                         landmarks_12, landmarks_4])

    return face_cheeks_color[0], face_cheeks_color[1], face_cheeks_color[2]


def get_face_hair_color(image_bgr):
    resized_image, dets, _ = ugt.crop_face(image_bgr, detector, 480, 352)
    hair_color = [-1, -1, -1]
    if dets != -1:
        image_rgb = ugc.switch_rgb(resized_image)

        """hair detection"""
        seq = net.process(resized_image)
        seq = np.sum(seq, axis=2)

        hair_mask = np.zeros((seq.shape[0], seq.shape[1]), dtype='uint8')
        for i in range(seq.shape[0]):
            for j in range(seq.shape[1]):
                if seq[i, j] != 743:
                    hair_mask[i, j] = 250

        if sum(sum(hair_mask)) > 0:
            hair_color = ugc.get_dominant_color_masked(resized_image, hair_mask)

    return hair_color[0], hair_color[1], hair_color[2], image_rgb


if __name__ == '__main__':
    for path_from_top, subdirs, files in dirs:
        for f in tqdm(files):
            if not (f.endswith('jpg') or (f.endswith('png'))):
                continue
            image_path = str(path_from_top) + '/' + str(f)

            output_dict = dict()
            output_dict['file_name'] = str(f)
            path_parts = path_from_top.split('/')
            output_dict['set_folder'] = path_parts[-2]
            output_dict['face_type'] = path_parts[-1]

            image = imread(image_path)
            image = util.img_as_ubyte(rescale(image, (0.5, 0.5, 1), anti_aliasing=False))
            imageBGR = ugc.switch_rgb(image)

            """get face landmarks"""
            landmarks, continue_ = get_landmarks(image, f)
            if continue_:
                continue

            """get face under eyes color"""
            output_dict['face_undereyes_color_h'], output_dict[
                'face_undereyes_color_s'], output_dict[
                'face_undereyes_color_v'] = get_face_undereyes_color(
                imageBGR,
                landmarks[1],
                landmarks[29],
                landmarks[15]
            )

            """get face forehead color"""
            output_dict['face_forehead_color_h'], output_dict['face_forehead_color_s'], output_dict[
                'face_forehead_color_v'] = get_face_forehead_color(
                imageBGR,
                landmarks
            )

            """get face cheeks color"""
            output_dict['face_cheeks_color_h'], output_dict['face_cheeks_color_s'], output_dict[
                'face_cheeks_color_v'] = get_face_cheeks_color(
                imageBGR,
                landmarks[1],
                landmarks[15],
                landmarks[12],
                landmarks[4]
            )

            """crop/resize image for hair detection"""
            output_dict['hair_color_h'], output_dict['hair_color_s'], output_dict[
                'hair_color_v'], imageRGB = get_face_hair_color(imageBGR)

            """append produced data"""
            print(output_dict)
            output_list.append(output_dict)


    output_df = pd.DataFrame(output_list)
    output_df.to_csv(save_folder+'output_colors_augmentation.csv')
    print(len(empty_faces))
