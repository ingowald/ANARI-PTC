# ======================================================================== #
# Copyright 2024++ Ingo Wald                                               #
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

cmake_minimum_required(VERSION 3.12)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(ANARI_PTC VERSION 1.0.0 LANGUAGES C CXX)

# ------------------------------------------------------------------
option(PTC_DISABLE_CUDA "Explicitly Disable CUDA Support" off)
if (PTC_DISABLE_CUDA)
  set(USE_CUDA OFF)
else()
  include(CheckLanguage)
  check_language(CUDA)
  if (DEFINED CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(USE_CUDA ON)
    if (NOT (DEFINED CMAKE_CUDA_ARCHITECTURES))
      if (NOT CMAKE_VERSION VERSION_LESS "3.17")
	set(CMAKE_CUDA_ARCHITECTURES OFF)
      endif()
    endif()
  else()
    set(USE_CUDA OFF)
  endif()
endif()

# ------------------------------------------------------------------
find_package(anari 0.8.0 REQUIRED)
# ------------------------------------------------------------------
find_package(MPI REQUIRED)
# ------------------------------------------------------------------

if (USE_CUDA)
  add_subdirectory(submodules/deepCompositing)
else()
  add_subdirectory(cpu_compositing)
endif()

if ((${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

add_subdirectory(ptc)
