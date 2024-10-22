// ======================================================================== //
// Copyright 2024 Ingo Wald                                                 //
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

// anari
#include <anari/anari.h>
#include <anari/anari_cpp/ext/linalg.h>
#include <anari/frontend/type_utility.h>
// helium
#include <helium/utility/IntrusivePtr.h>
// std
#include <mpi.h>
#include <functional>
#include <cstdlib>
#include <cstring>

namespace ptc {

  using FrameWrapperNotificationHandler = std::function<void(const void *)>;

  struct FrameWrapper : public helium::RefCounted
  {
    FrameWrapper(ANARIDevice d,
                 ANARIFrame f,
                 FrameWrapperNotificationHandler onObjectDestroy,
                 MPI_Comm = MPI_COMM_WORLD);
    virtual ~FrameWrapper() override;

    ANARIFrame handle() const;

    // Forward generic ANARI API calls to underlying ANARIFrame /////////////////
    void setParameter(const char *name, ANARIDataType type, const void *mem);
    void unsetParameter(const char *name);
    void unsetAllParameters();
    void *mapParameterArray1D(const char *name,
                              ANARIDataType dataType,
                              uint64_t numElements1,
                              uint64_t *elementStride);
    void *mapParameterArray2D(const char *name,
                              ANARIDataType dataType,
                              uint64_t numElements1,
                              uint64_t numElements2,
                              uint64_t *elementStride);
    void *mapParameterArray3D(const char *name,
                              ANARIDataType dataType,
                              uint64_t numElements1,
                              uint64_t numElements2,
                              uint64_t numElements3,
                              uint64_t *elementStride);
    void unmapParameterArray(const char *name);
    void commitParameters();
    void release();
    void retain();
    int getProperty(const char *name,
                    ANARIDataType type,
                    void *mem,
                    uint64_t size,
                    ANARIWaitMask mask);
    const void *frameBufferMap(const char *channel,
                               uint32_t *width,
                               uint32_t *height,
                               ANARIDataType *pixelType);
    void frameBufferUnmap(const char *channel);
    virtual void renderFrame() = 0;
    int frameReady(ANARIWaitMask);
    void discardFrame();
    /////////////////////////////////////////////////////////////////////////////

  protected:
    virtual void updateSize();
    virtual void composite() = 0;
    virtual void cleanup() = 0;

    FrameWrapperNotificationHandler m_onObjectDestroy;

    int m_rank{-1};

    ANARIDevice m_device{nullptr};
    ANARIFrame m_frame{nullptr};

    ANARIDataType m_newColorType{ANARI_UNKNOWN};
    ANARIDataType m_currentColorType{ANARI_UNKNOWN};
    anari::math::uint2 m_newSize{0u, 0u};
    anari::math::uint2 m_currentSize{0u, 0u};

    std::vector<float> m_depth;
    std::vector<uint8_t> m_color;
    uint32_t *d_color_in{nullptr};
    uint32_t *d_color_out{nullptr};
    float *d_depth{nullptr};
  };


  // Helper functions ///////////////////////////////////////////////////////////

  template <typename T>
  T bit_cast(ANARIDataType type, const void *mem)
  {
    if (sizeof(T) != anari::sizeOf(type))
      throw std::runtime_error("T size mismatch");
    T retval;
    memcpy(&retval, mem, sizeof(T));
    return retval;
  }



} // namespace ptc
