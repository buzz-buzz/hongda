from scipy.ndimage.filters import gaussian_filter
from moviepy.editor import VideoFileClip
from PIL import Image, ImageEnhance
from numpy import array
import cv2
import numpy as np
import os.path
import os
import dlib

nose_img_file = os.path.join(os.getcwd(), 'hongda/pasters/nose.jpg')
nose_paster = cv2.imread(nose_img_file)

face_detector = dlib.get_frontal_face_detector()
landmark_file_path = os.path.join(os.getcwd(), 'hongda/data/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(landmark_file_path)

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def rect_to_bounding_box(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


def sub_image(img, x, y, w, h):
    clip = sub_area(img, x, y, w, h)
    return img[clip[1]:clip[1] + clip[3], clip[0]:clip[0] + clip[2]]

def sub_area(img, x, y, w, h):
    print('image shape = ', img.shape)
    print(x, y, w, h)
    if x < 0:
        crop_x = -x
        x = 0
        w -= crop_x
    else:
        crop_x = 0

    if y < 0:
        print('y is less than zero!')
        crop_y = -y
        y = 0
        h -= crop_y
    else:
        crop_y = 0

    if w > img.shape[1] - x:
        w = img.shape[1] - x

    if h > img.shape[0] - y:
        print('h is overflow!')
        h = img.shape[0] - y

    crop_end_x = crop_x + w
    crop_end_y = crop_y + h
    print(x, y, w, h)
    return (x, y, w, h, crop_x, crop_y, crop_end_x, crop_end_y)


def paster_nose_to(img, paster, center_x, center_y, paste_to_width, paste_to_height):
    if paste_to_height == None:
        paste_to_height = int((paster.shape[0] / paster.shape[1]) * paste_to_width)
    paster = cv2.resize(paster, (paste_to_width, paste_to_height), interpolation=cv2.INTER_AREA)
    start_x = int(center_x - paste_to_width / 2)
    start_y = int(center_y - paste_to_height / 2)

    clip = sub_area(img, start_x, start_y, paste_to_width, paste_to_height)

    _, paster_mask = cv2.threshold(
        cv2.cvtColor(paster, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY_INV)
    paster = paster[clip[5]:clip[7], clip[4]:clip[6]]
    paster_mask = paster_mask[clip[5]:clip[7], clip[4]:clip[6]]
    inv_paster_mask = cv2.bitwise_not(paster_mask)
    masked_paster = cv2.bitwise_and(paster, paster, mask=paster_mask)

    nose_area = sub_image(img, start_x, start_y, paste_to_width, paste_to_height)
    masked_nose_area = cv2.bitwise_and(nose_area, nose_area, mask=inv_paster_mask)
    merged = cv2.add(masked_nose_area, masked_paster)

    img[clip[1]:clip[1] + clip[3],
    clip[0]:clip[0] + clip[2]] = merged

    return img


def paster_nose_2(img):
    scale = 200 / min(img.shape[1], img.shape[0])
    print('scale= ', scale)
    gray = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

    faces = face_detector(gray, 1)

    for i, face_rect in enumerate(faces):
        shape = predictor(gray, face_rect)
        shape = shape_to_np(shape)
        (x, y, w, h) = rect_to_bounding_box(face_rect)
        cv2.circle(img, (int(shape[30][0] / scale), int(shape[30][1] / scale)), 2, (255, 0, 0), -1)
        paster_nose_to(img, nose_paster, int(shape[30][0] / scale), int(shape[30][1] / scale), int((w / scale) * 1.25), None)

    return img