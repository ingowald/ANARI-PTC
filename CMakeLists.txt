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
  if (CMAKE_CUDA_COMPILER)
    message("#ptc: found CUDA. going to use it.")
    if (CMAKE_CUDA_ARCHITECTURES)
      # specified on cmdline - check if cmake can understand this
      if ((${CMAKE_VERSION} VERSION_LESS 3.24) AND
          ("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "all-major"))
        message(AUTHOR_WARNING "'all-major' cuda arch requested, but this cmake version doesn't have this. replacing with '70;80'")
        set(PTC_CUDA_ARCH_INIT "70;80")
      else()
        set(PTC_CUDA_ARCH_INIT "${CMAKE_CUDA_ARCHITECTURES}")
      endif()
    else()
      if (${CMAKE_VERSION} VERSION_LESS 3.24)
        set(PTC_CUDA_ARCH_INIT "70;80")
      else()
        #use 'all-major' by default, in case this is a CI build, where
        #there migth be no 'native' gpu installed
        set(PTC_CUDA_ARCH_INIT "all-major")
      endif()
    endif()
    set(CMAKE_CUDA_ARCHITECTURES
      "${PTC_CUDA_ARCH_INIT}" CACHE STRING
      "Which CUDA architecture to build for")
    enable_language(CUDA)
    set(USE_CUDA ON)
    message("#ptc: CUDA is on, with arch(s) ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    message("#ptc: did NOT find CUDA. disabling.")
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
