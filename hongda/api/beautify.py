from scipy.ndimage.filters import gaussian_filter
from moviepy.editor import VideoFileClip
from PIL import Image, ImageEnhance
from numpy import array
from moviepy import *
import moviepy.video.fx.all as vfx
import cv2
import numpy as np
import os.path
from api.video import compress_dimension_with_rotation_handled, convert_mp4_to_mov


def blur(image):
    return gaussian_filter(image.astype(float), sigma=2)


def contrast(image):
    c = ImageEnhance.Contrast(Image.fromarray(image)).enhance(1.2)
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
    img_small = cv2.resize(img, None, fx=1.0 / ds_factor, fy=1.0 / ds_factor, interpolation=cv2.INTER_AREA)
    num_repetitions = 10
    sigma_color = 5
    sigma_space = 7
    size = 5

    for i in range(num_repetitions):
        img_small = cv2.bilateralFilter(img_small, size, sigma_color, sigma_space)

    img_output = cv2.resize(img_small, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    dst = np.zeros(img_gray.shape)

    # Add the thick boundary lines to the image using 'AND' operator
    dst = cv2.bitwise_and(img_output, img_output, mask=mask)
    return dst


def beautify(image):
    # return cartoonize(bright(contrast(sharp(image))))
    return cartoonize(bright(image))


def recipe_cartoonize(videoFilePath):
    parsed = os.path.split(videoFilePath)
    cartoonized_video_file = os.path.join(parsed[0], 'c-' + os.path.splitext(parsed[1])[0] + '.mp4')
    d = compress_dimension_with_rotation_handled(videoFilePath)
    clip = VideoFileClip(videoFilePath, target_resolution=d)
    clip_processed = clip.fl_image(beautify)
    print("cartoonized video: {0}".format(cartoonized_video_file))
    clip_processed.write_videofile(cartoonized_video_file)
    convert_mp4_to_mov(cartoonized_video_file)

# clip = VideoFileClip(r"C:\Users\Jeff\AppData\Local\Temp\8318652456269066__90582751-0C2D-41BD-957F-4E111C01609A.mp4",
#                      target_resolution=(640, 480))
# # clip_blurred = clip.fl_image(blur)
# # clip_processed = clip.fl_image(contrast)
# # clip_processed = clip.fl_image(sharp)
# #clip_processed = clip.fl_image(bright)
# #clip_processed = clip.fl_image(cartoonize)
# clip_processed = clip.fl_image(beautify)
# # clip_processed = clip.fx(vfx.painting, saturation=1, black=0.006)
# clip_processed.write_videofile("blurred_video.mp4")
#
#
