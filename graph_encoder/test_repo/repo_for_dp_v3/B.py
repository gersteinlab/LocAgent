# B.py

from interface.C import C


class B(C):
    def __init__(self):
        self.inner_instance = None
        self.value = 10

    def method_b1(self):
        pass


def func_b2():
    pass


def helper():
    print("This is external_helper_function in B.py.")

    def func_b1():
        print("This is inner_helper_b1 in external_helper_function.")

    def func_b2():
        print("This is inner_helper_b2 in external_helper_function.")
        func_b1()

    func_b2()
