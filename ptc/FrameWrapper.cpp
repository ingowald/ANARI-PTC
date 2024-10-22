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

#include "FrameWrapper.h"
// std
#include <cstring>
#include <string_view>
// anari
#include <anari/frontend/type_utility.h>
#include <anari/anari_cpp.hpp>

namespace ptc {

// GPU kernels ////////////////////////////////////////////////////////////////

// FrameWrapper definitions ///////////////////////////////////////////////////

FrameWrapper::FrameWrapper(ANARIDevice d,
    ANARIFrame f,
    FrameWrapperNotificationHandler onObjectDestroy,
    MPI_Comm comm)
    : m_onObjectDestroy(onObjectDestroy),
      m_device(d),
      m_frame(f)
{
  MPI_Comm_rank(comm, &m_rank);
  anariRetain(m_device, m_device);
  anari::setParameter(m_device, m_frame, "channel.depth", ANARI_FLOAT32);
}

FrameWrapper::~FrameWrapper()
{
  anariRelease(m_device, m_frame);
  anariRelease(m_device, m_device);
  m_onObjectDestroy(this);
}

ANARIFrame FrameWrapper::handle() const
{
  return m_frame;
}

void FrameWrapper::setParameter(
    const char *_name, ANARIDataType type, const void *mem)
{
  std::string_view name = _name;

  if (type == ANARI_UINT32_VEC2 && name == "size")
    m_newSize = bit_cast<anari::math::uint2>(type, mem);
  else if (type == ANARI_DATA_TYPE && name == "channel.color")
    m_newColorType = bit_cast<ANARIDataType>(type, mem);
  else if (type == ANARI_DATA_TYPE && name == "channel.depth")
    return; // we don't want the app to turn off the depth channel

  anariSetParameter(m_device, m_frame, _name, type, mem);
}

void FrameWrapper::unsetParameter(const char *name)
{
  anariUnsetParameter(m_device, m_frame, name);
}

void FrameWrapper::unsetAllParameters()
{
  anariUnsetAllParameters(m_device, m_frame);
  anari::setParameter(m_device, m_frame, "channel.depth", ANARI_FLOAT32);
}

void *FrameWrapper::mapParameterArray1D(const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t *elementStride)
{
  return anariMapParameterArray1D(
      m_device, m_frame, name, dataType, numElements1, elementStride);
}

void *FrameWrapper::mapParameterArray2D(const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t *elementStride)
{
  return anariMapParameterArray2D(m_device,
      m_frame,
      name,
      dataType,
      numElements1,
      numElements2,
      elementStride);
}

void *FrameWrapper::mapParameterArray3D(const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t numElements3,
    uint64_t *elementStride)
{
  return anariMapParameterArray3D(m_device,
      m_frame,
      name,
      dataType,
      numElements1,
      numElements2,
      numElements3,
      elementStride);
}

void FrameWrapper::unmapParameterArray(const char *name)
{
  anariUnmapParameterArray(m_device, m_frame, name);
}

void FrameWrapper::commitParameters()
{
  updateSize();
  anariCommitParameters(m_device, m_frame);
}

void FrameWrapper::release()
{
  refDec();
}

void FrameWrapper::retain()
{
  refInc();
}

int FrameWrapper::getProperty(const char *name,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    ANARIWaitMask mask)
{
  return anariGetProperty(m_device, m_frame, name, type, mem, size, mask);
}

const void *FrameWrapper::frameBufferMap(const char *_channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  *width = m_currentSize.x;
  *height = m_currentSize.y;

  std::string_view channel = _channel;

  if (channel == "channel.color") {
    *pixelType = m_currentColorType;
    return m_color.data();
  } else if (channel == "channel.depth") {
    *pixelType = ANARI_FLOAT32;
    return m_depth.data();
  }

  *width = 0;
  *height = 0;
  *pixelType = ANARI_UNKNOWN;
  return nullptr;
}

void FrameWrapper::frameBufferUnmap(const char *channel)
{
  // no-op
}

int FrameWrapper::frameReady(ANARIWaitMask m)
{
  return anariFrameReady(m_device, m_frame, m);
}

void FrameWrapper::discardFrame()
{
  anariDiscardFrame(m_device, m_frame);
}

void FrameWrapper::updateSize()
{
  if (m_newSize == m_currentSize)
    return;

  cleanup();

  m_currentSize = m_newSize;
  m_currentColorType = m_newColorType;

  if (m_currentSize.x * m_currentSize.y == 0)
    return;

  if (m_currentColorType == ANARI_UNKNOWN)
    return;

  if (m_currentColorType == ANARI_FLOAT32_VEC4) {
    throw std::runtime_error(
        "support for FLOAT32_VEC4 color channel not implemented");
  }

  const auto &size = m_currentSize;

  m_color.resize(size.x * size.y * sizeof(anari::math::float4));
  m_depth.resize(size.x * size.y);

}

void FrameWrapper::renderFrame()
{
  anariRenderFrame(m_device, m_frame);
  anariFrameReady(m_device,m_frame,ANARI_WAIT);
  composite();
}

  
} // namespace ptc
