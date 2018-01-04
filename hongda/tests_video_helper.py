import os
from api.video import *
import unittest
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))

test_file = os.path.join(dir_path, 'tests/bplus.mp4')

class VideoHelperTest(unittest.TestCase):
    def test_convert_webm_to_mp4(self):
        mp4 = convert_webm_to_mp4(test_file)
        self.assertEqual(os.path.isfile(mp4), True)

if __name__ == '__main__':
    unittest.main(warnings='ignore')
