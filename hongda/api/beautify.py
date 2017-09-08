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
    return paster_nose((img))


def paster_effect_nose_2(img):
    return paster_nose_2((img))


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
nose_paster = cv2.imread(nose_img_file)

face_detector = dlib.get_frontal_face_detector()
landmark_file_path = os.path.join(os.getcwd(), 'hongda/data/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(landmark_file_path)

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


def paster_nose(img):
    if nose_cascade.empty():
        raise IOError(
            'Unable to load the nose classifier xml file: ' + nose_cascade_file)

    nose_paster = cv2.imread(nose_img_file)
    nose_paster_natural_height, node_paster_natural_width = nose_paster.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        roi_gray = gray[y:y + h, x:x + w]
        roi_img = img[y:y + h, x:x + w]

        nose_rects = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (x_nose, y_nose, w_nose, h_nose) in nose_rects:
            paster_width = int(w * 1.25)
            paster_height = int((nose_paster_natural_height / node_paster_natural_width) * paster_width)

            paster_start_y_from_face = int(y_nose - (paster_height - h_nose) / 2)
            paster_start_x_from_face = int(x_nose - (paster_width - w_nose) / 2)
            start_y_from_img = y + paster_start_y_from_face
            start_x_from_img = x + paster_start_x_from_face

            clip = sub_area(img, start_x_from_img, start_y_from_img, paster_width, paster_height)

            nose_area = sub_image(img, start_x_from_img, start_y_from_img, paster_width, paster_height)

            small_nose_paster = cv2.resize(
                nose_paster, (paster_width, paster_height), interpolation=cv2.INTER_AREA)
            cropped_nose_paster = small_nose_paster[clip[5]:clip[7], clip[4]:clip[6]]

            gray_cropped_nose_paster = cv2.cvtColor(cropped_nose_paster, cv2.COLOR_BGR2GRAY)

            _, nose_paster_mask = cv2.threshold(
                gray_cropped_nose_paster, 200, 255, cv2.THRESH_BINARY_INV)
            inv_nose_paster_mask = cv2.bitwise_not(nose_paster_mask)

            masked_nose_paster = cv2.bitwise_and(
                cropped_nose_paster, cropped_nose_paster, mask=nose_paster_mask)

            print('nose area shape = ', nose_area.shape)
            print('mask shape = ', inv_nose_paster_mask.shape)
            print('paster shape = ', small_nose_paster.shape)

            # cv2.imshow('nose_area', nose_area)
            # cv2.imshow('mask', inv_nose_paster_mask)
            # cv2.imshow('paster', small_nose_paster)

            try:
                masked_nose_area = cv2.bitwise_and(nose_area, nose_area, mask=inv_nose_paster_mask)

                merged_nose_area = cv2.add(masked_nose_paster, masked_nose_area)

                img[clip[1]:clip[1] + clip[3],
                clip[0]:clip[0] + clip[2]] = merged_nose_area
            except:
                return nose_area

    return img


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


def recipe_paster_nose_2(video_file_path):
    if not video_file_path:
        video_file_path = r'C:\Users\Jeff\AppData\Local\Temp\6082852567445745__AA8315D1-82EB-4206-B98F-5FE24AE7191F.MOV'
    parsed = os.path.split(video_file_path)
    pastered = os.path.join(
        parsed[0], 'n-' + os.path.splitext(parsed[1])[0] + '.mp4')
    d = compress_dimension_with_rotation_handled(video_file_path)
    clip = VideoFileClip(video_file_path, target_resolution=d)
    clip_processed = clip.fl_image(paster_effect_nose_2)
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
