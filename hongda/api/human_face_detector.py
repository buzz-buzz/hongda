from scipy.ndimage.filters import gaussian_filter
from moviepy.editor import VideoFileClip
from PIL import Image, ImageEnhance
from numpy import array
import cv2
import numpy as np
import os.path
import os
import dlib
from api.image_helper import *

__all__ = ['paste_face']

dir_path = os.path.dirname(os.path.realpath(__file__))

nose_img_file = os.path.realpath(os.path.join(dir_path, '../pasters/nose.jpg'))
nose_paster = cv2.imread(nose_img_file)

sun_glasses_file = os.path.realpath(os.path.join(dir_path, '../pasters/sunglasses.jpg'))
sun_glasses_paster = cv2.imread(sun_glasses_file)

moustache_file = os.path.realpath(os.path.join(dir_path, '../pasters/moustache.png'))
moustache_paster = cv2.imread(moustache_file)

face_detector = dlib.get_frontal_face_detector()
landmark_file_path = os.path.realpath(os.path.join(dir_path, '../data/shape_predictor_68_face_landmarks.dat'))
predictor = dlib.shape_predictor(landmark_file_path)


def paste_face(img, pasters):
    scale = 200 / min(img.shape[1], img.shape[0])
    gray = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray, 1)

    for i, face_rect in enumerate(faces):
        shape = predictor(gray, face_rect)
        shape = shape_to_np(shape)
        (x, y, w, h) = rect_to_bounding_box(face_rect)
        for paster, landmark_index, ratio, valign in [recipes[paster_name] for paster_name in pasters]:
            center_x = int(shape[landmark_index][0] / scale)
            center_y = int(shape[landmark_index][1] / scale)
            add_paster(img, paster, center_x, center_y, int(ratio * w / scale), None, valign)

    return img


recipe_nose = (nose_paster, 30, 1.25, None)
recipe_sun_glasses = (sun_glasses_paster, 27, 1, None)
recipe_moustache = (moustache_paster, 51, 0.4, 'bottom')

recipes = {
    'recipe_nose': recipe_nose,
    'recipe_sun_glasses': recipe_sun_glasses,
    'recipe_moustache': recipe_moustache
}