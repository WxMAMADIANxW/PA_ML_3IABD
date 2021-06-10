from ctypes import *

path_to_shared_library = "../CPPDLLForPython/cmake-build-debug/CPPDLLForPython.dll"

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    my_lib = cdll.LoadLibrary(path_to_shared_library)

    res = my_lib.toto()
    print(res)
