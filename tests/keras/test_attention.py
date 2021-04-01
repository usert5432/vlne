import unittest

import numpy as np
import tensorflow as tf

from lstm_ee.keras.models.models_atten import (
    DotAttention, slice_trasnpose, unslice_transpose
)

class TestAttention(unittest.TestCase):
    _test_atten = DotAttention(dropout = None)
    _null_atten = tf.keras.layers.Attention(dropout = None)

    def _compare_data(self, keys, values, queries):
        test = self._test_atten([queries, keys, values])
        null = self._null_atten([queries, values, keys])

        self.assertTrue(np.array_equal(test, null))

    def test_dot_simple(self):
        keys    = np.random.rand(1, 3, 3)
        values  = np.random.rand(1, 3, 1)
        queries = np.random.rand(1, 3, 3)

        self._compare_data(keys, values, queries)

    def test_dot_simple_batch(self):
        keys    = np.random.rand(10, 3, 3)
        values  = np.random.rand(10, 3, 1)
        queries = np.random.rand(10, 3, 3)

        self._compare_data(keys, values, queries)

    def test_dot_complex(self):
        keys    = np.random.rand(10, 11, 4)
        values  = np.random.rand(10, 11, 10)
        queries = np.random.rand(10, 11, 4)

        self._compare_data(keys, values, queries)

class TestSliceUnsliceTranspose(unittest.TestCase):

    def _test_slice_unslice(self, v, n):
        tv = slice_trasnpose(v, n)
        tv = unslice_transpose(tv, n)

        self.assertTrue(np.array_equal(tv, v))

    def test_simple(self):
        v = np.random.rand(1, 10, 100)
        n = 1
        self._test_slice_unslice(v, n)

    def test_multihead(self):
        v = np.random.rand(1, 10, 100)
        n = 2
        self._test_slice_unslice(v, n)

    def test_simple_batch_simple(self):
        v = np.random.rand(10, 20, 100)
        n = 1
        self._test_slice_unslice(v, n)

    def test_simple_batch_multihead(self):
        v = np.random.rand(10, 20, 100)
        n = 4
        self._test_slice_unslice(v, n)


