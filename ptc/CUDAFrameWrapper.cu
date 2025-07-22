// ======================================================================== //
// Copyright 2024++ Ingo Wald                                               //
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

#include "CUDAFrameWrapper.h"
// std
#include <cstring>
#include <string_view>
// anari
#include <anari/frontend/type_utility.h>
#include <anari/anari_cpp.hpp>

#include "cuda_helper.h"

namespace ptc {

  // GPU kernels ////////////////////////////////////////////////////////////////
  
  __global__ void writeFrags(dc::DeviceInterface di,
                             uint32_t size_x,
                             uint32_t size_y,
                             const float *d_depth,
                             const void *d_color,
                             ANARIDataType colorType)
  {
    const int ix = threadIdx.x + blockIdx.x * blockDim.x;
    const int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix >= size_x)
      return;
    if (iy >= size_y)
      return;

    const auto offset = ix + iy * size_x;
    const float z = d_depth[offset];

    if (colorType == ANARI_FLOAT32_VEC4) {
      const float *rgba = (const float *)d_color + (offset * 4);
#if 1
      const float r = rgba[0];
      const float g = rgba[1];
      const float b = rgba[2];
      const float a = rgba[3];
#else
      const float a = rgba[0];
      const float b = rgba[1];
      const float g = rgba[2];
      const float r = rgba[3];
#endif
      dc::Fragment frag(z, make_float3(r, g, b), a);
      di.write(ix, iy, frag);
    } else {
      const uint32_t rgba = *((const uint32_t *)d_color + offset);
      const float a = ((rgba >> 24) & 0xff) / 255.f;
      const float b = ((rgba >> 16) & 0xff) / 255.f;
      const float g = ((rgba >> 8) & 0xff) / 255.f;
      const float r = ((rgba >> 0) & 0xff) / 255.f;
      dc::Fragment frag(z, make_float3(r, g, b), a);
      di.write(ix, iy, frag);
    }
  }

  // FrameWrapper definitions ///////////////////////////////////////////////////

  CUDAFrameWrapper
  ::CUDAFrameWrapper(ANARIDevice d,
                     ANARIFrame f,
                     FrameWrapperNotificationHandler onObjectDestroy,
                     MPI_Comm comm)
    : FrameWrapper(d,f,onObjectDestroy,comm),
      m_deepComp(1, comm)
  {
    m_deepComp.affinitizeGPU();
  }

  CUDAFrameWrapper
  ::~CUDAFrameWrapper()
  {
    CUDAFrameWrapper::cleanup();
  }
  
  void CUDAFrameWrapper::updateSize()
  {
    if (m_newSize == m_currentSize && m_newColorType == m_currentColorType)
      return;
    FrameWrapper::updateSize();

    if (m_currentSize.x * m_currentSize.y == 0)
      return;

    if (m_currentColorType == ANARI_UNKNOWN)
      return;
    
    const auto &size = m_currentSize;
    
    m_deepComp.resize(size.x, size.y);

    size_t colorSize
      = (m_currentColorType == ANARI_FLOAT32_VEC4)
      ? sizeof(float4)
      : sizeof(uint32_t);
    cudaMalloc((void **)&d_color_in,  size.x * size.y * colorSize);
    cudaMalloc((void **)&d_color_out, size.x * size.y * colorSize);
    cudaMalloc((void **)&d_depth, size.x * size.y * sizeof(*d_depth));
    CUDA_SYNC_CHECK();
  }

  void CUDAFrameWrapper::composite()
  {
    dc::DeviceInterface dcDev = m_deepComp.prepare();

    anari::math::uint2 size;
    ANARIDataType ptType = ANARI_UNKNOWN;
    const float *depth
      = (const float *)anariMapFrame(m_device, m_frame, "channel.depth",
                                     &size.x, &size.y, &ptType);
    const void *color
      = (const void *)anariMapFrame(m_device, m_frame, "channel.color",
                                    &size.x, &size.y, &ptType);
    size_t colorSize
      = (m_currentColorType == ANARI_FLOAT32_VEC4)
      ? sizeof(float4)
      : sizeof(uint32_t);
    cudaMemcpy
      (d_depth, depth, size.x * size.y * sizeof(float),
       cudaMemcpyDefault);
    cudaMemcpy
      (d_color_in, color, size.x * size.y * colorSize,
       cudaMemcpyDefault);
    
    auto ngx = dc::divRoundUp(size.x, 16);
    auto ngy = dc::divRoundUp(size.y, 16);

    writeFrags<<<dim3(ngx, ngy), dim3(16, 16)>>>
      (m_deepComp.prepare(),
       size.x,
       size.y,
       d_depth,
       d_color_in,
       m_currentColorType);
    bool useFloat4 = (m_currentColorType == ANARI_FLOAT32_VEC4);
    m_deepComp.finish(m_rank == 0 ? d_color_out : nullptr,
                      useFloat4);
    cudaMemcpy(m_color, 
               d_color_out,
               size.x * size.y * colorSize,
               cudaMemcpyDefault);

    CUDA_SYNC_CHECK();

    anariUnmapFrame(m_device, m_frame, "channel.depth");
    anariUnmapFrame(m_device, m_frame, "channel.color");
  }

  void CUDAFrameWrapper::cleanup()
  {
    if (d_depth) 
      cudaFree(d_depth);
    if (d_color_in)
      cudaFree(d_color_in);
    if (d_color_out)
      cudaFree(d_color_out);

    CUDA_SYNC_CHECK();

    d_depth = nullptr;
    d_color_in = nullptr;
    d_color_out = nullptr;
  }

} // namespace ptc
