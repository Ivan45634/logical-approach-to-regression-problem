import ctypes
import os
from typing import List

import numpy

script_dir = os.path.dirname(os.path.realpath(__file__))
librunc = ctypes.CDLL(script_dir + "/librunc.so")

librunc.init_dualizer.restype = ctypes.c_void_p

librunc.release_dualizer.argtypes = [ctypes.c_void_p]

librunc.set_input_row.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                  numpy.ctypeslib.ndpointer(dtype=numpy.int32)]

librunc.enumerate_covers.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                     numpy.ctypeslib.ndpointer(dtype=numpy.int32)]
librunc.enumerate_covers.restype = ctypes.c_int


class RuncDualizer:

    def __init__(self):
        self.dualizer = librunc.init_dualizer()
        self.m = 0
        self.n = 0

    def add_input_row(self, columns_with_ones: List[int]):
        row = numpy.array(columns_with_ones, dtype=numpy.int32)
        n = numpy.max(row) + 1
        if self.n < n:
            self.n = n
        librunc.set_input_row(self.dualizer, self.m, len(row), row)
        self.m = self.m + 1

    def enumerate_covers(self, buffer_count: int = 128) -> List[numpy.array]:
        covers_buffer = numpy.zeros(buffer_count * (self.n + 1), dtype=numpy.int32)
        c = librunc.enumerate_covers(self.dualizer, len(covers_buffer), covers_buffer)

        offset = 0
        result = []
        for i in range(c):
            size = covers_buffer[offset]
            offset += 1
            cover = covers_buffer[offset:offset + size]
            offset += size
            result.append(cover)

        return result

    def __del__(self):
        librunc.release_dualizer(self.dualizer)


