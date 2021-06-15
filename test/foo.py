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
    def __init__(self, name):
        self.name_ = name

    def print(self, prefix):
        print("Python: ", prefix, self.name_)

    def name(self):
        return self.name_


def ShowBytes(bs):
    print("Python: ", bs)
    return bs
