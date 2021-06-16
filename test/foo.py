import numpy as np

num_pups = 5


def FooFunc():
    print("Python: Hi, Paw Patrol!")


def BarFunc(name):
    s = "Hi, " + name
    print("Python:", s)
    return s


class FooClass:
    prefix = "Paw Patrol"

    def __init__(self, name):
        self.name_ = name

    def name(self):
        return self.name_

    @staticmethod
    def Prefix():
        return FooClass.prefix


def DictKeys(d: dict):
    return list(d.keys())


def DictKeysAndValues(d: dict):
    return list(d.keys()), list(d.values())


def NdarrayAdd(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    z = x + y
    return z
