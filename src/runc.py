import ctypes
import os
from typing import List

import numpy

script_dir = os.path.dirname(os.path.realpath(__file__))
# librunc = ctypes.CDLL(script_dir + "/librunc.so")
librunc = ctypes.CDLL(script_dir + "/librunc.dylib")

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
        if len(columns_with_ones) == 0:
            return False
        row = numpy.array(columns_with_ones, dtype=numpy.int32)
        n = numpy.max(row) + 1
        if self.n < n:
            self.n = n
        librunc.set_input_row(self.dualizer, self.m, len(row), row)
        self.m = self.m + 1
        return True

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


def test_enumerate():
    # Example of using RuncDualizer to small input matrix
    runc = RuncDualizer()
    runc.add_input_row([0, 7, 10, 16])
    runc.add_input_row([0, 1, 7])
    runc.add_input_row([1, 7, 6])

    while True:
        covers = runc.enumerate_covers()
        if len(covers) == 0:
            print("End")
            break

        for cover in covers:
            print(cover)


def test_enumerate_large():
    # Example of using RuncDualizer to small input matrix
    runc = RuncDualizer()

    for i in range(180):
        runc.add_input_row([i, i + 1])

    for c in range(20):
        covers = runc.enumerate_covers()
        if len(covers) == 0:
            print("End")
            break

        for cover in covers:
            print(cover)


# test_enumerate_large()

def test_enumerate_from_file():
    # Example of using RuncDualizer for matrix loaded from file.
    # In this example we are enumerating covers by parts with many calls of enumerate_covers
    runc = RuncDualizer()
    with open('/home/user/dual/src/tests/data/6_6_4_0.hg', 'r') as f:
        for l in f.readlines():
            runc.add_input_row([int(a) for a in l.split(' ')])

    while True:
        covers = runc.enumerate_covers(10)
        print(len(covers))

        if len(covers) == 0:
            print("End")
            break
