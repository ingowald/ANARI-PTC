# ======================================================================== #
# Copyright 2024__ Ingo Wald                                               #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

## PTC device target ##

if (USE_CUDA)
  project(anari_library_ptc LANGUAGES C CXX CUDA)
else()
  project(anari_library_ptc LANGUAGES C CXX)
endif()

if (USE_CUDA)
  project_add_library(MODULE
    PTCDevice.cpp
    PTCLibrary.cpp
    FrameWrapper.cpp
    CUDAFrameWrapper.cu)
  set_source_files_properties(CUDAFrameWrapper.cu
    PROPERTIES COMPILE_FLAGS "--expt-relaxed-constexpr"
    )
  project_link_libraries(anari::helium deepCompositing)
  project_compile_definitions(PRIVATE -DPTC_CUDA=1)
  project_include_directories(PRIVATE ${PROJECT_SOURCE_DIR}/../submodules/deepCompositing)
else()
  project_add_library(MODULE
    PTCDevice.cpp
    PTCLibrary.cpp
    FrameWrapper.cpp
    CPUFrameWrapper.cpp)
  project_link_libraries(anari::helium cpu_compositing)
endif()


include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME} EXPORT_MACRO_NAME "PTC_EXPORT")
project_include_directories(PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

## Installation ##

include(GNUInstallDirs)

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
