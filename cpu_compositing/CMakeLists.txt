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

# ------------------------------------------------------------------
find_package(TBB REQUIRED)
# ------------------------------------------------------------------

add_library(cpu_compositing STATIC
  CPUCompositor.h
  CPUCompositor.cpp
  )

target_link_libraries(cpu_compositing
  MPI::MPI_CXX
  anari::helium
  tbb
  )
target_include_directories(cpu_compositing
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  )

set_property(TARGET cpu_compositing PROPERTY POSITION_INDEPENDENT_CODE ON)

