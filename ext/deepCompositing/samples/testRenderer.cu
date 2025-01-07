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

#include "testRenderer.h"
#include "owl/common/math/vec.h"
#include "owl/common/math/box.h"

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
    // idx.x = (idx.x * ((1<<11)-1) + 0x12345);
    // idx.y = (idx.y * ((1<<13)-1) + 0x23456);
    // idx.z = (idx.z * ((1<<15)-1) + 0x34567);

    // idx.x = (idx.x * ((1<<13)-1) + 0x123457);
    // idx.y = (idx.y * ((1<<15)-1) + 0x234567);
    // idx.z = (idx.z * ((1<<17)-1) + 0x345677);

    // idx.x = (idx.x * ((1<<13)-1) + 0x123457);
    // idx.y = (idx.y * ((1<<15)-1) + 0x234567);
    // idx.z = (idx.z * ((1<<17)-1) + 0x345677);

    // return idx.x ^ idx.y ^ idx.z;
  }

  inline __device__
  bool isMine(vec3i idx,
              int mpiRank,
              int mpiSize)
  {
    return (scramble(idx)%mpiSize) == mpiRank;
  }

  inline __device__
  bool intersects(const box3f &box,
                  const vec3f &org,
                  const vec3f &dir,
                  float &t)
  {
#if 0
    int N = 40;
    int hits = 0;
    for (int i=0;i<N;i++)
      for (int j=0;j<N;j++) {
        vec3f _dir = dir;
        vec3f dir = _dir + vec3f(i*.0001f,j*.00001f,0.f);
        vec3f lo = (box.lower - org) * rcp(dir);
        vec3f hi = (box.upper - org) * rcp(dir);
        vec3f nr = min(lo,hi);
        vec3f fr = max(lo,hi);
        t = max(0.f,reduce_max(nr));
        if (reduce_min(fr) >= t)
          hits++;
      }
    return hits > N*N / 2;
#else
    vec3f lo = (box.lower - org) * rcp(dir);
    vec3f hi = (box.upper - org) * rcp(dir);
    vec3f nr = min(lo,hi);
    vec3f fr = max(lo,hi);
    t = max(0.f,reduce_max(nr));
    return reduce_min(fr) >= t;
#endif
  }

  // ==================================================================
  // first variant - use fixd num frags...
  // ==================================================================
  __global__
  void renderKernel(int testCubeRes,
                    dc::DeviceInterface deepFB,
                    const Camera camera,
                    int mpiRank, int mpiSize)
  {
    int pixelX = threadIdx.x + blockIdx.x * blockDim.x;
    int pixelY = threadIdx.y + blockIdx.y * blockDim.y;
    if (pixelX >= deepFB.getSize().x) return;
    if (pixelY >= deepFB.getSize().y) return;

    const vec3f org = camera.org;
    const vec3f dir
      = camera.dir_00
      + float(pixelX) * camera.dir_du
      + float(pixelY) * camera.dir_dv;

    if ((pixelY%mpiSize) == mpiRank) {
      const float t = pixelY / (float)(deepFB.getSize().y);
      const vec3f bgColor
        = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f)
        +         t *vec3f(0.5f, 0.7f, 1.0f);
      deepFB.write(pixelX,pixelY,dc::Fragment(1e10f,bgColor,1.f));
    }

    for (int iz=0;iz<testCubeRes;iz++)
      for (int iy=0;iy<testCubeRes;iy++)
        for (int ix=0;ix<testCubeRes;ix++) {
          if (!isMine({ix,iy,iz},mpiRank,mpiSize))
            continue;

          box3f box;
          box.lower = (vec3f(ix,iy,iz)+.1f) * 1.f/testCubeRes;
          box.upper = (vec3f(ix,iy,iz)+.9f) * 1.f/testCubeRes;
          float t;
          if (intersects(box,org,dir,t)) {
            vec3f color = owl::randomColor(12+mpiRank);
            deepFB.write(pixelX,pixelY,dc::Fragment(t,color,.8f));
          }
        }
  }

  void renderFragments(int testCubeRes,
                       const dc::DeviceInterface &deepFB,
                       const Camera &camera,
                       int mpiRank, int mpiSize)
  {
    assert(deepFB.fbSize.x > 0);
    assert(deepFB.fbSize.y > 0);

    const int blockSize = 8;
    dim3 numBlocks = { (unsigned)divRoundUp(deepFB.fbSize.x,blockSize),
                       (unsigned)divRoundUp(deepFB.fbSize.y,blockSize),
                       (unsigned)1 };

    // PRINT(deepFB.fbSize.x);
    // PRINT(deepFB.fbSize.y);
    // PRINT(deepFB.fragments);
    // PRINT(deepFB.counters);

    // PRINT((vec3i)numBlocks);
    // PRINT((vec3i)blockSize);
    renderKernel<<<numBlocks,dim3(blockSize,blockSize,1)>>>
      (testCubeRes,deepFB,camera,mpiRank,mpiSize);
    CUDA_SYNC_CHECK();
  }

}


