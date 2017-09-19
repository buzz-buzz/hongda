import numpy as np

__all__ = ['shape_to_np', 'rect_to_bounding_box', 'sub_image']

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
