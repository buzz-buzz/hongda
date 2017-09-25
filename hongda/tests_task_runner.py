import os
from api.task_runner import *
import unittest
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))

test_file = os.path.join(dir_path, 'tasks.txt')

def task_1():
    print('task - 1 started...')
    with open(test_file, 'a') as f:
        f.write('t')
    print('task - 1 ended')
    # sys.stdout.flush()


def task_2():
    print('task - 2 started...')
    with open(test_file, 'a') as f:
        f.write('t')
    print('task -2 ended')
    # sys.stdout.flush()


class ImageHelperTest(unittest.TestCase):
    def test_parallel_run(self):
        parallel_run([task_1, task_2], [(), ()])

        with open(test_file, 'r') as f:
            text = f.read()

        os.remove(test_file)
        self.assertEqual(text, 'tt')

    def parallel_run_lambda(self):
        parallel_run([lambda : task_1(), lambda : task_2()])

        with open(test_file, 'r') as f:
            text = f.read()

        os.remove(test_file)
        self.assertEqual(text, 'tt')


if __name__ == '__main__':
    unittest.main(warnings='ignore')
