import numpy as np

num_pups = 5


def FooFunc():
    print("Hi, Raw Patrol!")


def BarFunc(name):
    s = "Hi, " + name
    print("Python: ", s)
    return s


class FooClass:
    def __init__(self):
        self.name_ = "Rubble"

    def print(self):
        print("Python: ", self.name_)

    def name(self):
        return self.name_


class BarClass:
    prefix = "Hi,"

    def __init__(self, name):
        self.name_ = name

    def print(self, postfilx):
        print("Python: ", BarClass.prefix, self.name_, postfilx)

    def name(self):
        return self.name_

    @staticmethod
    def Prefix():
        return BarClass.prefix


def ShowBytes(bs):
    print("Python: ", bs)
    return bs


def DictKeys(d: dict):
    return list(d.keys())


def DictKeysAndValues(d: dict):
    return list(d.keys()), list(d.values())


def GetNdarray():
    x = np.zeros((3, 2), dtype=np.float32)
    k = 0
    for i in range(3):
        for j in range(2):
            x[i, j] = k
            k += 1
    return x


def NdarrayAdd(x: np.ndarray, c) -> np.ndarray:
    y = x + c
    return y


def CvMatToNdarray(x: np.ndarray) -> np.ndarray:
    y = x + 10
    return y
