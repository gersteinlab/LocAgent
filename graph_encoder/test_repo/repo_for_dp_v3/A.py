# A.py

from interface.C import C
from B import B
# from B import *


def decorator(func):
    def wrapper(*args, **kwargs):
        print("Decorator in A.py is called.")
        return func(*args, **kwargs)
    return wrapper


async def async_func():
    print("This is an async function in A.py.")


class A(C):
    def __init__(self):
        self.inner_instance = None
        self.value = 0
        self.method_a2()

    def method_a1(self):
        from B import helper as ehf

        print("This is main_method_a1 of MainClassA.")
        self.value += 1
        self.inner_instance = self.InnerA()
        self.inner_instance.inner_a()
        ehf()

    @decorator
    def method_a2(self):
        print("This is main_method_a2 of MainClassA.")

        async def inner_async():
            await async_func()

        import asyncio
        asyncio.run(inner_async())

    class InnerA:
        def inner_a(self):
            print("This is inner_method_a of InnerClassA.")
            B().method_b1()
