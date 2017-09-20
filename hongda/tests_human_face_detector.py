import unittest
import os
from api.human_face_detector import *
import cv2
from skimage.measure import compare_mse as im_diff


dir_path = os.path.dirname(os.path.realpath(__file__))

img_path = os.path.join(dir_path, 'tests/xinwen.jpg')

class HumanFaceDetectorTest(unittest.TestCase):

    def test_add_nose(self):
        img = cv2.imread(img_path)
        nosed_img_path = os.path.join(dir_path, 'tests/n-xinwen.jpg')
        paste_face(img, [recipe_nose])
        expected_nosed_img = cv2.imread(nosed_img_path)
        diff = im_diff(img, expected_nosed_img)
        print('diff = ', diff)
        self.assertLess(diff, 1)

    def test_add_sun_glasses(self):
        img = cv2.imread(img_path)
        sun_glasses_img_path = os.path.join(dir_path, 'tests/e-xinwen.jpg')
        paste_face(img, [recipe_sun_glasses])
        expected = cv2.imread(sun_glasses_img_path)

        diff = im_diff(img, expected)
        print('diff == ', diff)

        if diff > 1:
            cv2.imshow('actual', img)
            cv2.imshow('expected', expected)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self.assertLess(diff, 1)

    def test_add_sun_glasses_and_moustache(self):
        img = cv2.imread(img_path)
        pasted_path = os.path.join(dir_path, 'tests/em-xinwen.jpg')
        paste_face(img, [recipe_sun_glasses, recipe_moustache])
        expected = cv2.imread(pasted_path)

        diff = im_diff(img, expected)
        print('diff === ', diff)
        self.assertLess(diff, 1)

if __name__ == '__main__':
    unittest.main(warnings='ignore')
