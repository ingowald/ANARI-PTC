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

#include <mpi.h>
#include "testRenderer.h"
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION 1
# include "stb/stb_image.h"
# include "stb/stb_image_write.h"

namespace dctest {

  Camera camera;
  // vec2i  fbSize = { 320,200 };
  vec2i  fbSize = { 1024,1024 };
  
  extern "C" int main(int ac, char **av)
  {
    std::string outFileName = "";
    int testCubeRes = 5;
    for (int i=1;i<ac;i++) {
      const std::string arg = av[i];
      if (arg == "-n")
        testCubeRes = std::stol(av[++i]);
      else if (arg == "-o")
        outFileName = av[++i];
      else
        throw std::runtime_error("unknown cmdline arg "+arg);
    }
    
    // ------------------------------------------------------------------
    // setup
    // ------------------------------------------------------------------
    /* set up a default camera pointing to a cube of (0,0,0)-(1,1,1) */
    vec3f lookAt = { .5,.5,.5 };
    vec3f lookFrom = { -1,-2,-3 };
    vec3f lookUp   = { 0,1,0 };
    float zoom = 2.4f;

    vec3f dir = normalize(lookAt-lookFrom);
    vec3f du  = normalize(cross(dir,lookUp));
    vec3f dv  = normalize(cross(du,dir));
    
    camera.org = lookFrom;
    camera.dir_00 = zoom*dir - .5f*du - .5f*dv;
    camera.dir_du = du * 1.f/fbSize.x;
    camera.dir_dv = dv * 1.f/fbSize.y;

    /* initialize MPI */
    MPI_Init(&ac,&av);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    if (rank == 0) /* to force an output over "Invalid Cookie" error message" */
      PING;
    /* create our deep compositor */
    dc::Compositor compositor;
    compositor.affinitizeGPU();

    compositor.resize(fbSize.x,fbSize.y);
    dc::DeviceInterface interface = compositor.prepare();
    // ------------------------------------------------------------------
    // render one frame
    // ------------------------------------------------------------------
    renderFragments(testCubeRes, interface, camera, rank, size);

    // ------------------------------------------------------------------
    // composite frame
    // ------------------------------------------------------------------
    uint32_t *fbPointer = nullptr;
    uint32_t *fbHostPointer = nullptr;
    if (rank == 0) {
      cudaMalloc(&fbPointer,fbSize.x*fbSize.y*sizeof(*fbPointer));
      fbHostPointer = (uint32_t *)malloc(fbSize.x*fbSize.y*sizeof(*fbPointer));
    }

    if (outFileName != "") {
      compositor.dumpFileName = outFileName+"_"+std::to_string(rank)+".dcfb";
      // compositor.save(outFileName);
    }
    

    compositor.finish(fbPointer);

    // ------------------------------------------------------------------
    // on root: write to file
    // ------------------------------------------------------------------
    if (rank == 0) {
      cudaMemcpy(fbHostPointer,fbPointer,fbSize.x*fbSize.y*sizeof(*fbPointer),
                 cudaMemcpyDeviceToHost);
      const std::string fileName
        = (outFileName=="")
        ? "testDeepOffline.png"
        : outFileName+".png";
      stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                     fbHostPointer,fbSize.x*sizeof(uint32_t));
      cudaFree(fbPointer);
      free(fbHostPointer);
    }

    // ------------------------------------------------------------------
    // done rendering
    // ------------------------------------------------------------------
    MPI_Finalize();
    return 0;
  }
}
