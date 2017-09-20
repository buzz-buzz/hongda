import unittest
import os
from api.human_face_detector import *
import cv2
from skimage.measure import compare_mse as im_diff


dir_path = os.path.dirname(os.path.realpath(__file__))

class HumanFaceDetectorTest(unittest.TestCase):

    def test_add_nose(self):
        img_path = os.path.join(dir_path, 'tests/xinwen.jpg')
        nosed_img_path = os.path.join(dir_path, 'tests/n-xinwen.jpg')
        img = cv2.imread(img_path)
        paste_face(img, [recipe_nose])
        expected_nosed_img = cv2.imread(nosed_img_path)
        diff = im_diff(img, expected_nosed_img)
        print('diff = ', diff)
        self.assertLess(diff, 1)

    def test_add_sun_glasses(self):
        img_path = os.path.join(dir_path, 'tests/xinwen.jpg')
        sun_glasses_img_path = os.path.join(dir_path, 'tests/e-xinwen.jpg')
        img = cv2.imread(img_path)
        paste_face(img, [recipe_sun_glasses])
        expected = cv2.imread(sun_glasses_img_path)

        diff = im_diff(img, expected)
        print('diff == ', diff)
        self.assertLess(diff, 1)

if __name__ == '__main__':
    unittest.main(warnings='ignore')
