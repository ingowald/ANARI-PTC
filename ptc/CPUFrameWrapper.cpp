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

#include "CPUFrameWrapper.h"
// std
#include <cstring>
#include <string_view>
// anari
#include <anari/frontend/type_utility.h>
#include <anari/anari_cpp.hpp>

namespace ptc {

  // GPU kernels ////////////////////////////////////////////////////////////////
  
  // __global__ void writeFrags(dc::DeviceInterface di,
  //                            uint32_t size_x,
  //                            uint32_t size_y,
  //                            const float *d_depth,
  //                            const void *d_color,
  //                            ANARIDataType colorType)
  // {
  //   const int ix = threadIdx.x + blockIdx.x * blockDim.x;
  //   const int iy = threadIdx.y + blockIdx.y * blockDim.y;
  //   if (ix >= size_x)
  //     return;
  //   if (iy >= size_y)
  //     return;

  //   const auto offset = ix + iy * size_x;
  //   const float z = d_depth[offset];

  //   if (colorType == ANARI_FLOAT32_VEC4) {
  //     const float *rgba = (const float *)d_color + (offset * 4);
  //     const float a = rgba[0];
  //     const float b = rgba[1];
  //     const float g = rgba[2];
  //     const float r = rgba[3];
  //     dc::Fragment frag(z, make_float3(r, g, b), a);
  //     di.write(ix, iy, frag);
  //   } else {
  //     const uint32_t rgba = *((const uint32_t *)d_color + offset);
  //     const float a = ((rgba >> 24) & 0xff) / 255.f;
  //     const float b = ((rgba >> 16) & 0xff) / 255.f;
  //     const float g = ((rgba >> 8) & 0xff) / 255.f;
  //     const float r = ((rgba >> 0) & 0xff) / 255.f;
  //     dc::Fragment frag(z, make_float3(r, g, b), a);
  //     di.write(ix, iy, frag);
  //   }
  // }

  // FrameWrapper definitions ///////////////////////////////////////////////////

  CPUFrameWrapper
  ::CPUFrameWrapper(ANARIDevice d,
                     ANARIFrame f,
                     FrameWrapperNotificationHandler onObjectDestroy,
                     MPI_Comm comm)
    : FrameWrapper(d,f,onObjectDestroy,comm),
      compositor(comm)
  {
  }

  CPUFrameWrapper
  ::~CPUFrameWrapper()
  {
    CPUFrameWrapper::cleanup();
  }
  
  void CPUFrameWrapper::updateSize()
  {
    if (m_newSize == m_currentSize)
      return;
    FrameWrapper::updateSize();

    if (m_currentSize.x * m_currentSize.y == 0)
      return;

    if (m_currentColorType == ANARI_UNKNOWN)
      return;

    if (m_currentColorType == ANARI_FLOAT32_VEC4) {
      throw std::runtime_error
        ("support for FLOAT32_VEC4 color channel not implemented");
    }

    const auto &size = m_currentSize;

    compositor.resize(size.x, size.y);

    m_color.resize(size.x*size.y);
    m_depth.resize(size.x*size.y);
  }

  void CPUFrameWrapper::composite()
  {
    anari::math::uint2 size;
    ANARIDataType ptType = ANARI_UNKNOWN;
    const float *depth
      = (const float *)anariMapFrame(m_device, m_frame, "channel.depth",
                                     &size.x, &size.y, &ptType);
    const uint32_t *color
      = (const uint32_t *)anariMapFrame(m_device, m_frame, "channel.color",
                                        &size.x, &size.y, &ptType);

    compositor.run((uint32_t*)m_color.data(),
                   (float *)m_depth.data(),
                   color,depth);

    anariUnmapFrame(m_device, m_frame, "channel.depth");
    anariUnmapFrame(m_device, m_frame, "channel.color");
  }

  void CPUFrameWrapper::cleanup()
  {
  }

} // namespace ptc
