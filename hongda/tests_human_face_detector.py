import unittest
import os
# from api.human_face_detector import *
import cv2

dir_path = os.path.dirname(os.path.realpath(__file__))

class HumanFaceDetectorTest(unittest.TestCase):

    def test_shap_to_np(self):

    
    def test_human_face_detect(self):
        img_path = os.path.join(dir_path, 'tests/xinwen.jpg')
        img = cv2.imread(img_path)
        cv2.imshow( 'output', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    unittest.main(warnings='ignore')
