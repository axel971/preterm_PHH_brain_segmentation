# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


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
CMAKE_COMMAND = /usr/bin/cmake3

# The command to remove a file.
RM = /usr/bin/cmake3 -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/axel/dev/neonatal_brain_segmentation/source/preprocessing/relabeling_8VOIs

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/axel/dev/neonatal_brain_segmentation/source/preprocessing/relabeling_8VOIs/build

# Include any dependencies generated for this target.
include CMakeFiles/relabeling_8VOIs.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/relabeling_8VOIs.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/relabeling_8VOIs.dir/flags.make

CMakeFiles/relabeling_8VOIs.dir/source/relabeling_8VOIs.o: CMakeFiles/relabeling_8VOIs.dir/flags.make
CMakeFiles/relabeling_8VOIs.dir/source/relabeling_8VOIs.o: ../source/relabeling_8VOIs.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/axel/dev/neonatal_brain_segmentation/source/preprocessing/relabeling_8VOIs/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/relabeling_8VOIs.dir/source/relabeling_8VOIs.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/relabeling_8VOIs.dir/source/relabeling_8VOIs.o -c /home/axel/dev/neonatal_brain_segmentation/source/preprocessing/relabeling_8VOIs/source/relabeling_8VOIs.cpp

CMakeFiles/relabeling_8VOIs.dir/source/relabeling_8VOIs.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/relabeling_8VOIs.dir/source/relabeling_8VOIs.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/axel/dev/neonatal_brain_segmentation/source/preprocessing/relabeling_8VOIs/source/relabeling_8VOIs.cpp > CMakeFiles/relabeling_8VOIs.dir/source/relabeling_8VOIs.i

CMakeFiles/relabeling_8VOIs.dir/source/relabeling_8VOIs.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/relabeling_8VOIs.dir/source/relabeling_8VOIs.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/axel/dev/neonatal_brain_segmentation/source/preprocessing/relabeling_8VOIs/source/relabeling_8VOIs.cpp -o CMakeFiles/relabeling_8VOIs.dir/source/relabeling_8VOIs.s

# Object files for target relabeling_8VOIs
relabeling_8VOIs_OBJECTS = \
"CMakeFiles/relabeling_8VOIs.dir/source/relabeling_8VOIs.o"

# External object files for target relabeling_8VOIs
relabeling_8VOIs_EXTERNAL_OBJECTS =

relabeling_8VOIs: CMakeFiles/relabeling_8VOIs.dir/source/relabeling_8VOIs.o
relabeling_8VOIs: CMakeFiles/relabeling_8VOIs.dir/build.make
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkdouble-conversion-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitksys-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkvnl_algo-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkvnl-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkv3p_netlib-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitknetlib-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkvcl-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKCommon-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkNetlibSlatec-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKStatistics-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKTransform-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKMesh-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkzlib-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKMetaIO-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKSpatialObjects-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKPath-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKLabelMap-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKQuadEdgeMesh-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOImageBase-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKOptimizers-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKPolynomials-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKBiasCorrection-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKDICOMParser-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKEXPAT-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkgdcmDICT-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkgdcmMSFF-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKznz-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKniftiio-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKgiftiio-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkhdf5_cpp.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkhdf5.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOBMP-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOBioRad-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOBruker-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOCSV-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOGDCM-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOIPL-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOGE-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOGIPL-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOHDF5-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkjpeg-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOJPEG-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkopenjpeg-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOJPEG2000-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitktiff-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOTIFF-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOLSM-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkminc2-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMINC-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMRC-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMeshBase-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMeshBYU-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMeshFreeSurfer-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMeshGifti-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMeshOBJ-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMeshOFF-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMeshVTK-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMeta-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIONIFTI-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKNrrdIO-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIONRRD-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkpng-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOPNG-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOSiemens-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOXML-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOSpatialObjects-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOStimulate-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKTransformFactory-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOTransformBase-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOTransformHDF5-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOTransformInsightLegacy-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOTransformMatlab-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOVTK-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKKLMRegionGrowing-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitklbfgs-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKOptimizersv4-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKTestKernel-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKVTK-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKVideoCore-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKVideoIO-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKWatersheds-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkopenjpeg-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkminc2-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOIPL-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOXML-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkhdf5_cpp.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkhdf5.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOTransformBase-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKTransformFactory-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKOptimizers-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitklbfgs-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOBMP-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOGDCM-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkgdcmMSFF-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkgdcmDICT-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkgdcmIOD-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkgdcmDSED-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkgdcmCommon-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkgdcmjpeg8-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkgdcmjpeg12-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkgdcmjpeg16-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkgdcmopenjp2-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkgdcmcharls-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkgdcmuuid-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOGIPL-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOJPEG-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOTIFF-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitktiff-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkjpeg-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMeshBYU-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMeshFreeSurfer-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMeshGifti-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKgiftiio-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKEXPAT-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMeshOBJ-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMeshOFF-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMeshVTK-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMeshBase-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKQuadEdgeMesh-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOMeta-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKMetaIO-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIONIFTI-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKniftiio-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKznz-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIONRRD-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKNrrdIO-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOPNG-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkpng-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkzlib-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOVTK-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKIOImageBase-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKVideoCore-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKStatistics-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkNetlibSlatec-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKSpatialObjects-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKMesh-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKTransform-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKPath-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKCommon-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkdouble-conversion-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitksys-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libITKVNLInstantiation-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkvnl_algo-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkvnl-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkv3p_netlib-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitknetlib-5.0.a
relabeling_8VOIs: /cm/shared/apps/itk/5.0/lib/libitkvcl-5.0.a
relabeling_8VOIs: CMakeFiles/relabeling_8VOIs.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/axel/dev/neonatal_brain_segmentation/source/preprocessing/relabeling_8VOIs/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable relabeling_8VOIs"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/relabeling_8VOIs.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/relabeling_8VOIs.dir/build: relabeling_8VOIs

.PHONY : CMakeFiles/relabeling_8VOIs.dir/build

CMakeFiles/relabeling_8VOIs.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/relabeling_8VOIs.dir/cmake_clean.cmake
.PHONY : CMakeFiles/relabeling_8VOIs.dir/clean

CMakeFiles/relabeling_8VOIs.dir/depend:
	cd /home/axel/dev/neonatal_brain_segmentation/source/preprocessing/relabeling_8VOIs/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/axel/dev/neonatal_brain_segmentation/source/preprocessing/relabeling_8VOIs /home/axel/dev/neonatal_brain_segmentation/source/preprocessing/relabeling_8VOIs /home/axel/dev/neonatal_brain_segmentation/source/preprocessing/relabeling_8VOIs/build /home/axel/dev/neonatal_brain_segmentation/source/preprocessing/relabeling_8VOIs/build /home/axel/dev/neonatal_brain_segmentation/source/preprocessing/relabeling_8VOIs/build/CMakeFiles/relabeling_8VOIs.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/relabeling_8VOIs.dir/depend

