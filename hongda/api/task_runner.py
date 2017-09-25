import logging
import multiprocessing
from multiprocessing import Pool

__all__ = ["parallel_run", "run_sequence"]


def parallel_run(tasks, args):
    print('len = ', len(tasks))
    multiprocessing.log_to_stderr(logging.INFO)
    pool = Pool(len(tasks))

    for index, t in enumerate(tasks):
        pool.apply_async(t, args=args[index])

    pool.close()
    pool.join()


def run_sequence(tasks):
    for t in tasks:
        t()
