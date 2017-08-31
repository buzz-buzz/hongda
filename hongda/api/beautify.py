from scipy.ndimage.filters import gaussian_filter
from moviepy.editor import VideoFileClip
from PIL import Image, ImageEnhance
from numpy import array
import cv2
import numpy as np
import os.path
from api.video import compress_dimension_with_rotation_handled, convert_mp4_to_mov
import os


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


def paster_effect(img):
    # return paster_nose(bright(img))
    # return paster_mouthache(paster_glasses(bright(img)))
    return paster_glasses(img)


def paster_effect_nose(img):
    return paster_nose(bright(img))


face_cascade_file = os.path.join(
    os.getcwd(), 'hongda/cascade_files/haarcascade_frontalface_alt.xml')
face_cascade = cv2.CascadeClassifier(face_cascade_file)
eye_cascade_file = os.path.join(
    os.getcwd(), 'hongda/cascade_files/haarcascade_eye.xml')
eye_cascade = cv2.CascadeClassifier(eye_cascade_file)
moustache_file = os.path.join(os.getcwd(), 'hongda/pasters/moustache.png')
sunglasses_file = os.path.join(os.getcwd(), 'hongda/pasters/sunglasses.png')

nose_cascade_file = os.path.join(
    os.getcwd(), 'hongda/cascade_files/haarcascade_mcs_nose.xml')
nose_cascade = cv2.CascadeClassifier(nose_cascade_file)
nose_img_file = os.path.join(os.getcwd(), 'hongda/pasters/nose.jpg')

def paster_glasses(img):
    if face_cascade.empty():
        raise IOError(
            'Unable to load the face cascade classifier xml file: ' + face_cascade_file)
    if eye_cascade.empty():
        raise IOError(
            'Unable to load the eye cascade classifier xml file:' + eye_cascade_file)

    sunglasses_img = cv2.imread(sunglasses_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    centers = []
    edges = []
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)

        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            center = (int(x_eye + 0.5 * w_eye), int(y_eye + 0.5 * h_eye))
            centers.append((x + center[0], y + center[1]))
            edges.append((x + x_eye, y + y_eye, w_eye, h_eye))

    if len(centers) <= 1 or len(edges) <= 1:
        return img

    try:
        x1 = edges[0][0] if edges[0][0] < edges[1][0] else edges[1][0]
        x1 -= 15
        x2 = edges[1][0] + \
            edges[1][2] if edges[0][0] < edges[1][0] else edges[0][0] + edges[0][2]
        x2 += 15
        y1 = edges[0][1]
        sunglasses_width = x2 - x1
        overlay_img = np.ones(img.shape, np.uint8) * 255
        h, w = sunglasses_img.shape[:2]
        scaling_factor = sunglasses_width / w
        overlay_sunglasses = cv2.resize(
            sunglasses_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        h, w = overlay_sunglasses.shape[:2]
        overlay_img[y1:y1 + h, x1:x1 + w] = overlay_sunglasses

        # create mask
        gray_sunglasses = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(gray_sunglasses, 222, 255, cv2.THRESH_BINARY)

        mask_inv = cv2.bitwise_not(mask)
        temp = cv2.bitwise_and(img, img, mask=mask)
        temp2 = cv2.bitwise_and(overlay_img, overlay_img, mask=mask_inv)
        final_img = cv2.add(temp, temp2)

        return final_img
        # return img
    except Exception as err:
        print(err)
        return img


def paster_mouthache(img):
    cascade_file = os.path.join(
        os.getcwd(), 'hongda/cascade_files/haarcascade_mcs_mouth.xml')
    moustache = cv2.CascadeClassifier(cascade_file)
    if moustache.empty():
        raise IOError(
            'Unable to load the moustache classifier xml file: ' + cascade_file)

    moustache_mask = cv2.imread(moustache_file)
    h_mask, w_mask = moustache_mask.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x_face, y_face, w_face, h_face) in faces:
        cv2.rectangle(img, (x_face, y_face), (x_face + w_face,
                                              y_face + h_face), (255, 255, 0), 3)
        roi_gray = gray[y_face:y_face + h_face, x_face:x_face + w_face]
        mouth_rects = moustache.detectMultiScale(roi_gray, 1.3, 5)
        roi_img = img[y_face:y_face + h_face, x_face:x_face + w_face]
        for (x, y, w, h) in mouth_rects:
            cv2.rectangle(roi_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

        for (x, y, w, h) in mouth_rects:
            width = int(w * 1.3)
            height = int((h_mask / w_mask) * width)
            start_y = y_face + y - (height - 10)
            start_x = x_face + int(x - (width - w) / 2)
            img_roi = img[y_face + start_y:y_face + start_y +
                          height, x_face + start_x:x_face + start_x + width]
            moustache_mask_small = cv2.resize(
                moustache_mask, (width, height), interpolation=cv2.INTER_AREA)
            gray_mask_small = cv2.cvtColor(
                moustache_mask_small, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(
                gray_mask_small, 200, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)
            masked_mouth = cv2.bitwise_and(
                moustache_mask_small, moustache_mask_small, mask=mask)
            masked_img = cv2.bitwise_and(img_roi, img_roi, mask=mask_inv)
            img[y_face + start_y:y_face + start_y + height, x_face + start_x:x_face + start_x + width] = cv2.add(
                masked_mouth, masked_img)

    return img


def paster_nose(img):
    if nose_cascade.empty():
        raise IOError(
            'Unable to load the nose classifier xml file: ' + nose_cascade_file)

    nose_img = cv2.imread(nose_img_file)
    h_nose_img, w_nose_img = nose_img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_img = img[y:y + h, x:x + w]

        nose_rects = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (x_nose, y_nose, w_nose, h_nose) in nose_rects:
            width = int(w * 1.25)
            height = int((h_nose_img / w_nose_img) * width)
            start_y = int(y_nose - (height - h_nose) / 2)
            start_x = int(x_nose - (width - w_nose) / 2)
            nose_area = img[y + start_y:y + start_y +
                            height, x + start_x:x + start_x + width]
            nose_img_small = cv2.resize(
                nose_img, (width, height), interpolation=cv2.INTER_AREA)
            gray_mask_small = cv2.cvtColor(nose_img_small, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(
                gray_mask_small, 80, 255, cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)
            masked_nose = cv2.bitwise_and(
                nose_img_small, nose_img_small, mask=mask)
            masked_img = cv2.bitwise_and(nose_area, nose_area, mask=mask_inv)
            img[y + start_y:y + start_y + height, x + start_x:x +
                start_x + width] = cv2.add(masked_nose, masked_img)
            # img[y + start_y:y + start_y + height, x + start_x:x + start_x + width] = masked_nose

    return img


def beautify(image):
    # return cartoonize(bright(contrast(sharp(image))))
    return cartoonize(bright(image))


def recipe_cartoonize(videoFilePath):
    parsed = os.path.split(videoFilePath)
    cartoonized_video_file = os.path.join(
        parsed[0], 'c-' + os.path.splitext(parsed[1])[0] + '.mp4')
    d = compress_dimension_with_rotation_handled(videoFilePath)
    clip = VideoFileClip(videoFilePath, target_resolution=d)
    clip_processed = clip.fl_image(beautify)
    print("cartoonized video: {0}".format(cartoonized_video_file))
    clip_processed.write_videofile(cartoonized_video_file)
    convert_mp4_to_mov(cartoonized_video_file)


def recipe_paster(video_file_path):
    if not video_file_path:
        video_file_path = r'C:\Users\Jeff\AppData\Local\Temp\6082852567445745__AA8315D1-82EB-4206-B98F-5FE24AE7191F.MOV'
    parsed = os.path.split(video_file_path)
    pastered = os.path.join(
        parsed[0], 'p-' + os.path.splitext(parsed[1])[0] + '.mp4')
    d = compress_dimension_with_rotation_handled(video_file_path)
    clip = VideoFileClip(video_file_path, target_resolution=d)
    clip_processed = clip.fl_image(paster_effect)
    clip_processed.write_videofile(pastered)
    convert_mp4_to_mov(pastered)


def recipe_paster_nose(video_file_path):
    if not video_file_path:
        video_file_path = r'C:\Users\Jeff\AppData\Local\Temp\6082852567445745__AA8315D1-82EB-4206-B98F-5FE24AE7191F.MOV'
    parsed = os.path.split(video_file_path)
    pastered = os.path.join(
        parsed[0], 'n-' + os.path.splitext(parsed[1])[0] + '.mp4')
    d = compress_dimension_with_rotation_handled(video_file_path)
    clip = VideoFileClip(video_file_path, target_resolution=d)
    clip_processed = clip.fl_image(paster_effect_nose)
    clip_processed.write_videofile(pastered)
    convert_mp4_to_mov(pastered)

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
