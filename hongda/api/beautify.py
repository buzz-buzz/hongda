from scipy.ndimage.filters import gaussian_filter
from moviepy.editor import VideoFileClip
from PIL import Image, ImageEnhance
from numpy import array
import cv2
import numpy as np
import os.path
from api.video import compress_dimension_with_rotation_handled, convert_mp4_to_mov
import os
import dlib
from api.human_face_detector import *


def blur(image):
    return gaussian_filter(image.astype(float), sigma=2)


def contrast(image):
    c = ImageEnhance.Contrast(Image.fromarray(image)).enhance(1.6)
    return array(c)


def sharp(image):
    shaprness = ImageEnhance.Sharpness(Image.fromarray(image)).enhance(7.0)
    return array(shaprness)


def bright(image):
    brightness = ImageEnhance.Brightness(Image.fromarray(image))
    r = brightness.enhance(1.5)
    return array(r)


def cartoonize(img, ds_factor=2, sketch_mode=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.medianBlur(img_gray, 9)

    # Detect edges in the image and threshold it
    edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)
    ret, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)
    if sketch_mode:
        img_sketch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        kernel = np.ones((3, 3), np.uint8)
        img_eroded = cv2.erode(img_sketch, kernel, iterations=1)
        return cv2.medianBlur(img_eroded, 5)
        # return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Resize the image to a smaller size for faster computation
    img_small = cv2.resize(img, None, fx=1.0 / ds_factor,
                           fy=1.0 / ds_factor, interpolation=cv2.INTER_AREA)
    num_repetitions = 10
    sigma_color = 5
    sigma_space = 7
    size = 5

    for i in range(num_repetitions):
        img_small = cv2.bilateralFilter(
            img_small, size, sigma_color, sigma_space)

    img_output = cv2.resize(img_small, None, fx=ds_factor,
                            fy=ds_factor, interpolation=cv2.INTER_AREA)
    dst = np.zeros(img_gray.shape)

    # Add the thick boundary lines to the image using 'AND' operator
    dst = cv2.bitwise_and(img_output, img_output, mask=mask)
    return dst

def paste_video(video_path, recipes=['recipe_nose']):
    parsed = os.path.split(video_path)
    pastered = os.path.join(
        parsed[0],
        'n-' + os.path.splitext(parsed[1])[0] + '.mp4'
    )

    d = compress_dimension_with_rotation_handled(video_path)
    clip = VideoFileClip(video_path, target_resolution=d)
    paster_recipes = generate_paster(recipes)
    clip_processed = clip.fl_image(paster_recipes)
    clip_processed.write_videofile(pastered)
    convert_mp4_to_mov(pastered)

def generate_paster(recipes):
    return lambda img: paste_face(img, recipes)