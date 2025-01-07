// ======================================================================== //
// Copyright 2018-2020 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "owl/common/math/vec.h"
#include "deepCompositing.h"

/*! a little test renderer that uses the deepCompositing library to
    render some NxNxN semi-transparent cubes in MPI, with differnt
    ranks owning differnet cubes */
namespace dctestImage {
  using namespace owl::common;
 
  struct Camera {
    vec3f org, dir_00, dir_du, dir_dv;
  };

  /*! the main (second) render pass: produce the actual fragments, and
      write then into the pre-allocated memory */
  void renderFragments(const dc::DeviceInterface &fs,
                       const Camera &camera, int mpiRank, int mpiSize,
                       const char *imgDir);
}

  
