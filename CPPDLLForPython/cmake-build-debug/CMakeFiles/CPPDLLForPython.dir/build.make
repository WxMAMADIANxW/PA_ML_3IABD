# CMAKE generated file: DO NOT EDIT!
# Generated by "NMake Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

!IF "$(OS)" == "Windows_NT"
NULL=
!ELSE
NULL=nul
!ENDIF
SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\Users\Mamadian\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\211.7142.21\bin\cmake\win\bin\cmake.exe

# The command to remove a file.
RM = C:\Users\Mamadian\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\211.7142.21\bin\cmake\win\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\Mamadian\Documents\ESGI\MachineLearning\PA_ML_3IABD\CPPDLLForPython

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\Mamadian\Documents\ESGI\MachineLearning\PA_ML_3IABD\CPPDLLForPython\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles\CPPDLLForPython.dir\depend.make

# Include the progress variables for this target.
include CMakeFiles\CPPDLLForPython.dir\progress.make

# Include the compile flags for this target's objects.
include CMakeFiles\CPPDLLForPython.dir\flags.make

CMakeFiles\CPPDLLForPython.dir\library.cpp.obj: CMakeFiles\CPPDLLForPython.dir\flags.make
CMakeFiles\CPPDLLForPython.dir\library.cpp.obj: ..\library.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\Mamadian\Documents\ESGI\MachineLearning\PA_ML_3IABD\CPPDLLForPython\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CPPDLLForPython.dir/library.cpp.obj"
	C:\PROGRA~2\MICROS~2\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoCMakeFiles\CPPDLLForPython.dir\library.cpp.obj /FdCMakeFiles\CPPDLLForPython.dir\ /FS -c C:\Users\Mamadian\Documents\ESGI\MachineLearning\PA_ML_3IABD\CPPDLLForPython\library.cpp
<<

CMakeFiles\CPPDLLForPython.dir\library.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CPPDLLForPython.dir/library.cpp.i"
	C:\PROGRA~2\MICROS~2\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe > CMakeFiles\CPPDLLForPython.dir\library.cpp.i @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\Mamadian\Documents\ESGI\MachineLearning\PA_ML_3IABD\CPPDLLForPython\library.cpp
<<

CMakeFiles\CPPDLLForPython.dir\library.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CPPDLLForPython.dir/library.cpp.s"
	C:\PROGRA~2\MICROS~2\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\cl.exe @<<
 /nologo /TP $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) /FoNUL /FAs /FaCMakeFiles\CPPDLLForPython.dir\library.cpp.s /c C:\Users\Mamadian\Documents\ESGI\MachineLearning\PA_ML_3IABD\CPPDLLForPython\library.cpp
<<

# Object files for target CPPDLLForPython
CPPDLLForPython_OBJECTS = \
"CMakeFiles\CPPDLLForPython.dir\library.cpp.obj"

# External object files for target CPPDLLForPython
CPPDLLForPython_EXTERNAL_OBJECTS =

CPPDLLForPython.dll: CMakeFiles\CPPDLLForPython.dir\library.cpp.obj
CPPDLLForPython.dll: CMakeFiles\CPPDLLForPython.dir\build.make
CPPDLLForPython.dll: CMakeFiles\CPPDLLForPython.dir\objects1.rsp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\Mamadian\Documents\ESGI\MachineLearning\PA_ML_3IABD\CPPDLLForPython\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library CPPDLLForPython.dll"
	C:\Users\Mamadian\AppData\Local\JetBrains\Toolbox\apps\CLion\ch-0\211.7142.21\bin\cmake\win\bin\cmake.exe -E vs_link_dll --intdir=CMakeFiles\CPPDLLForPython.dir --rc=C:\PROGRA~2\WI3CF2~1\10\bin\100190~1.0\x86\rc.exe --mt=C:\PROGRA~2\WI3CF2~1\10\bin\100190~1.0\x86\mt.exe --manifests -- C:\PROGRA~2\MICROS~2\2019\BUILDT~1\VC\Tools\MSVC\1429~1.300\bin\Hostx86\x64\link.exe /nologo @CMakeFiles\CPPDLLForPython.dir\objects1.rsp @<<
 /out:CPPDLLForPython.dll /implib:CPPDLLForPython.lib /pdb:C:\Users\Mamadian\Documents\ESGI\MachineLearning\PA_ML_3IABD\CPPDLLForPython\cmake-build-debug\CPPDLLForPython.pdb /dll /version:0.0 /machine:x64 /debug /INCREMENTAL  kernel32.lib user32.lib gdi32.lib winspool.lib shell32.lib ole32.lib oleaut32.lib uuid.lib comdlg32.lib advapi32.lib  
<<

# Rule to build all files generated by this target.
CMakeFiles\CPPDLLForPython.dir\build: CPPDLLForPython.dll

.PHONY : CMakeFiles\CPPDLLForPython.dir\build

CMakeFiles\CPPDLLForPython.dir\clean:
	$(CMAKE_COMMAND) -P CMakeFiles\CPPDLLForPython.dir\cmake_clean.cmake
.PHONY : CMakeFiles\CPPDLLForPython.dir\clean

CMakeFiles\CPPDLLForPython.dir\depend:
	$(CMAKE_COMMAND) -E cmake_depends "NMake Makefiles" C:\Users\Mamadian\Documents\ESGI\MachineLearning\PA_ML_3IABD\CPPDLLForPython C:\Users\Mamadian\Documents\ESGI\MachineLearning\PA_ML_3IABD\CPPDLLForPython C:\Users\Mamadian\Documents\ESGI\MachineLearning\PA_ML_3IABD\CPPDLLForPython\cmake-build-debug C:\Users\Mamadian\Documents\ESGI\MachineLearning\PA_ML_3IABD\CPPDLLForPython\cmake-build-debug C:\Users\Mamadian\Documents\ESGI\MachineLearning\PA_ML_3IABD\CPPDLLForPython\cmake-build-debug\CMakeFiles\CPPDLLForPython.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles\CPPDLLForPython.dir\depend

