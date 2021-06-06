from ctypes import *

path_to_shared_library = "../CPPDLLForPython/cmake-build-debug/CPPDLLForPython.dll"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    my_lib = cdll.LoadLibrary(path_to_shared_library)

    my_lib.random.argtypes = [c_float, c_float]
    my_lib.random.restypes = c_float

    res = my_lib.random(-1.0, 1.0)
    print(res)
