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
    anariFrameReady(m_device, m_frame, ANARI_WAIT);
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
