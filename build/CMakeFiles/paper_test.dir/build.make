# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

NVCC=/usr/local/cuda/bin/nvcc


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wp/LevelDB-CUDA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wp/LevelDB-CUDA/build

# Include any dependencies generated for this target.
include CMakeFiles/paper_test.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/paper_test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/paper_test.dir/flags.make

CMakeFiles/paper_test.dir/util/testharness.cc.o: CMakeFiles/paper_test.dir/flags.make
CMakeFiles/paper_test.dir/util/testharness.cc.o: ../util/testharness.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wp/LevelDB-CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/paper_test.dir/util/testharness.cc.o"
	$(NVCC)  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/paper_test.dir/util/testharness.cc.o -c /home/wp/LevelDB-CUDA/util/testharness.cc
	$(NVCC) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/paper_test.dir/util/decode_kv.cu.o -c /home/wp/LevelDB-CUDA/cuda/decode_kv.cu

CMakeFiles/paper_test.dir/util/testharness.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/paper_test.dir/util/testharness.cc.i"
	$(NVCC) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wp/LevelDB-CUDA/util/testharness.cc > CMakeFiles/paper_test.dir/util/testharness.cc.i

CMakeFiles/paper_test.dir/util/testharness.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/paper_test.dir/util/testharness.cc.s"
	$(NVCC) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wp/LevelDB-CUDA/util/testharness.cc -o CMakeFiles/paper_test.dir/util/testharness.cc.s

CMakeFiles/paper_test.dir/util/testutil.cc.o: CMakeFiles/paper_test.dir/flags.make
CMakeFiles/paper_test.dir/util/testutil.cc.o: ../util/testutil.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wp/LevelDB-CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/paper_test.dir/util/testutil.cc.o"
	$(NVCC)  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/paper_test.dir/util/testutil.cc.o -c /home/wp/LevelDB-CUDA/util/testutil.cc

CMakeFiles/paper_test.dir/util/testutil.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/paper_test.dir/util/testutil.cc.i"
	$(NVCC) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wp/LevelDB-CUDA/util/testutil.cc > CMakeFiles/paper_test.dir/util/testutil.cc.i

CMakeFiles/paper_test.dir/util/testutil.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/paper_test.dir/util/testutil.cc.s"
	$(NVCC) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wp/LevelDB-CUDA/util/testutil.cc -o CMakeFiles/paper_test.dir/util/testutil.cc.s

CMakeFiles/paper_test.dir/db/paper_test.cc.o: CMakeFiles/paper_test.dir/flags.make
CMakeFiles/paper_test.dir/db/paper_test.cc.o: ../db/paper_test.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wp/LevelDB-CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/paper_test.dir/db/paper_test.cc.o"
	$(NVCC)  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/paper_test.dir/db/paper_test.cc.o -c /home/wp/LevelDB-CUDA/db/paper_test.cc

CMakeFiles/paper_test.dir/db/paper_test.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/paper_test.dir/db/paper_test.cc.i"
	$(NVCC) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wp/LevelDB-CUDA/db/paper_test.cc > CMakeFiles/paper_test.dir/db/paper_test.cc.i

CMakeFiles/paper_test.dir/db/paper_test.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/paper_test.dir/db/paper_test.cc.s"
	$(NVCC) $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wp/LevelDB-CUDA/db/paper_test.cc -o CMakeFiles/paper_test.dir/db/paper_test.cc.s

# Object files for target paper_test
paper_test_OBJECTS = \
"CMakeFiles/paper_test.dir/util/testharness.cc.o" \
"CMakeFiles/paper_test.dir/util/testutil.cc.o" \
"CMakeFiles/paper_test.dir/db/paper_test.cc.o"

# External object files for target paper_test
paper_test_EXTERNAL_OBJECTS =

paper_test: CMakeFiles/paper_test.dir/util/testharness.cc.o
paper_test: CMakeFiles/paper_test.dir/util/testutil.cc.o
paper_test: CMakeFiles/paper_test.dir/db/paper_test.cc.o
paper_test: CMakeFiles/paper_test.dir/build.make
paper_test: libleveldb.a
paper_test: CMakeFiles/paper_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wp/LevelDB-CUDA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable paper_test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/paper_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/paper_test.dir/build: paper_test

.PHONY : CMakeFiles/paper_test.dir/build

CMakeFiles/paper_test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/paper_test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/paper_test.dir/clean

CMakeFiles/paper_test.dir/depend:
	cd /home/wp/LevelDB-CUDA/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wp/LevelDB-CUDA /home/wp/LevelDB-CUDA /home/wp/LevelDB-CUDA/build /home/wp/LevelDB-CUDA/build /home/wp/LevelDB-CUDA/build/CMakeFiles/paper_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/paper_test.dir/depend

