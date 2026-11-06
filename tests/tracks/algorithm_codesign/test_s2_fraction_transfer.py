"""Regression for large rational IPC only; no scientific evaluator/input."""
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from multiprocessing import get_context
import pickle
import sys
import unittest
from trottertracks.algorithm_codesign.parent_experiment.design import integer_io_limit


def echo(value):
    return value


class ExactTransfer(unittest.TestCase):
    def test_large_fraction_round_trip_and_restore(self):
        original=sys.get_int_max_str_digits()
        value=Fraction(1,10**5000+1)
        self.assertGreater(original,0)
        if original < 5000:
            with self.assertRaises(ValueError): pickle.dumps(value)
        # Parent serializes jobs and unpickles results. Forked children inherit
        # this private-process conversion policy, including their result pickle.
        with integer_io_limit(1_000_000), ProcessPoolExecutor(max_workers=1,mp_context=get_context('fork')) as pool:
            self.assertEqual(pool.submit(echo,value).result(),value)
        self.assertEqual(sys.get_int_max_str_digits(),original)


if __name__=='__main__': unittest.main()
