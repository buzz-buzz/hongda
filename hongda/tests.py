import unittest
import os
from api.text_matcher import *

dir_path = os.path.dirname(os.path.realpath(__file__))

class TextMatcherTest(unittest.TestCase):
    
    def test_text_match(self):
        score = text_match('Hello World', 'Hello World')
        self.assertEqual(score, 1)

    def test_get_vtt_content(self):
        expected_vtt_path = os.path.join(dir_path, './tests/exp-vtt.vtt')
        content = get_vtt_content(expected_vtt_path)
        self.assertEqual(content, 'We never really grow up, we only learn how to act in public.')
    
    def test_compare_2_vtt(self):
        expected_vtt_path = os.path.join(dir_path, 'tests/exp-vtt.vtt')
        actual_vtt_path = os.path.join(dir_path, 'tests/vtt.vtt')
            
        score = vtt_content_match(expected_vtt_path, actual_vtt_path)
        self.assertEqual(score, 0.32894736842105265)
        
if __name__ == '__main__':
    unittest.main(warnings='ignore')
