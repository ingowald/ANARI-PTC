// ======================================================================== //
// Copyright 2018-2024 Ingo Wald                                            //
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

#include "testRenderer.h"
#include "owl/common/math/vec.h"
#include "owl/common/math/box.h"
#include <fstream>

#define CUDA_SYNC_CHECK()                                       \
  {                                                             \
    cudaDeviceSynchronize();                                    \
    cudaError_t rc = cudaGetLastError();                        \
    if (rc != cudaSuccess) {                                    \
      fprintf(stderr, "error (%s: line %d): %s\n",              \
              __FILE__, __LINE__, cudaGetErrorString(rc));      \
      throw std::runtime_error("fatal cuda error");             \
    }                                                           \
  }



namespace dctest {
  using namespace owl::common;

  inline __device__ uint32_t scramble(vec3i idx)
  {
    enum { FNV_PRIME = 0x01000193 };
    uint32_t hash = 0x1234567;
    hash = hash * FNV_PRIME ^ idx.x;
    hash = hash * FNV_PRIME ^ idx.y;
    hash = hash * FNV_PRIME ^ idx.z;
    return hash;
  }

  inline __device__
  bool isMine(vec3i idx,
              int mpiRank,
              int mpiSize)
  {
    return (scramble(idx)%mpiSize) == mpiRank;
  }

  // ==================================================================
  // first variant - use fixd num frags...
  // ==================================================================
  void genBoxes(const std::string &fileName,
                int testCubeRes,
                int mpiRank, int mpiSize,
                float relSize,
                float alpha)
  {
    std::ofstream mat((fileName+".mat").c_str());
    mat << "newmtl default" << std::endl;
    vec3f color = owl::common::randomColor(3+mpiRank);
    mat << "Kd " << color.x << " " << color.y << " " << color.z << std::endl;
    mat << "d " << alpha << std::endl;
    
    std::ofstream obj((fileName+".obj").c_str());
    obj << "mtllib " << fileName << ".mat" << std::endl;
    for (int iz=0;iz<testCubeRes;iz++)
      for (int iy=0;iy<testCubeRes;iy++)
        for (int ix=0;ix<testCubeRes;ix++) {
          if (!isMine(vec3i{ix,iy,iz},mpiRank,mpiSize))
            continue;

          box3f box;
          float spaceOnSide = (1.f-relSize)/2.f;
          box.lower = (vec3f(ix,iy,iz)+spaceOnSide) * 1.f/testCubeRes - .5f;
          box.upper = (vec3f(ix,iy,iz)+(1.f-spaceOnSide)) * 1.f/testCubeRes - .5f;

          for (int iiz=0;iiz<2;iiz++)
            for (int iiy=0;iiy<2;iiy++)
              for (int iix=0;iix<2;iix++) {
                vec3f v(iix?box.lower.x:box.upper.x,
                        iiy?box.lower.y:box.upper.y,
                        iiz?box.lower.z:box.upper.z);
                obj << "v " << v.x << " " << v.y << " " << v.z << std::endl;
              }
          int indices[] = {0,1,3, 2,0,3,
            5,7,6, 5,6,4,
            0,4,5, 0,5,1,
            2,3,7, 2,7,6,
            1,5,7, 1,7,3,
            4,0,2, 4,2,6
          };
          for (int i=0;i<12;i++) {
            vec3i idx(indices[3*i+0] - 8,
                      indices[3*i+1] - 8,
                      indices[3*i+2] - 8);
            obj << "f " << idx.x << " " << idx.y << " " << idx.z << std::endl;
          }
        }
    obj << "done writing" << std::endl;
  }
  
  void genTestData(int ac, char **av)
  {
    std::string outFileName;
    int numRanks = 4;
    int testCubeRes = 4;
    float relSize = .8f;
    float alpha   = 1.f;
    for (int i=1;i<ac;i++) {
      std::string arg = av[i];
      if (arg == "-o")
        outFileName = av[++i];
      else if (arg == "-n")
        numRanks = std::stoi(av[++i]);
      else if (arg == "-a" || arg == "--alpha")
        alpha = std::stof(av[++i]);
      else 
        throw std::runtime_error("un-recognized cmdline argument '"+arg+"'");
    }
    if (outFileName.empty()) {
      std::cout << "dcGenTestData -o /path/outFileBase -n <numRanksToGenDataFor> -r <numCubesInEachDimension> -a alpha" << std::endl;
      exit(1);
    }
    for (int r=0;r<numRanks;r++) {
      std::cout << "generating data for rank " << r << " / " << numRanks << std::endl;
      genBoxes(outFileName+"."+std::to_string(r),
               testCubeRes,r,numRanks,relSize,alpha);
    }
  }
}

int main(int ac, char **av)
{
  dctest::genTestData(ac,av);
  return 0;
}

