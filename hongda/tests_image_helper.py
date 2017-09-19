import os
from api.image_helper import *
import unittest
import cv2
import time
from skimage.measure import compare_mse as im_diff

dir_path = os.path.dirname(os.path.realpath(__file__))


class ImageHelperTest(unittest.TestCase):
    def x_shape_to_np(self):
        shape = None
        np = shape_to_np(shape)

        self.assertEqual(np, [])

    def test_rect_to_bound_box(self):
        class Rect():
            def left(self):
                return 1

            def top(self):
                return 2

            def right(self):
                return 3

            def bottom(self):
                return 4

        rect = Rect()
        bb = rect_to_bounding_box(rect)
        self.assertEqual(bb, (1, 2, 2, 2))

    def test_sub_image(self):
        image_path = os.path.join(dir_path, './tests/xinwen.jpg')
        sub_image_path = os.path.join(dir_path, './tests/sub-xinwen.jpg')
        img = cv2.imread(image_path)
        cv2.imshow('Test', img)

        sub_img = sub_image(img, 0, 0, 100, 100)
        expected_sub_img = cv2.imread(sub_image_path)
        cv2.imshow('Test1', sub_img)

        diff = im_diff(sub_img, expected_sub_img)
        print('diff = ', diff)
        self.assertLess(diff, 1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    unittest.main(warnings='ignore')
