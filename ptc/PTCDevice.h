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

// helium
#include "helium/BaseDevice.h"
// std
#include <vector>

#include "FrameWrapper.h"

namespace ptc {

struct PTCDevice : public anari::DeviceImpl, helium::ParameterizedObject
{
  /////////////////////////////////////////////////////////////////////////////
  // Main interface to accepting API calls
  /////////////////////////////////////////////////////////////////////////////

  // Object creation //////////////////////////////////////////////////////////

  // Implement anariNewArray1D()
  ANARIArray1D newArray1D(const void *appMemory,
      ANARIMemoryDeleter deleter,
      const void *userdata,
      ANARIDataType,
      uint64_t numItems1) override;

  // Implement anariNewArray2D()
  ANARIArray2D newArray2D(const void *appMemory,
      ANARIMemoryDeleter deleter,
      const void *userdata,
      ANARIDataType,
      uint64_t numItems1,
      uint64_t numItems2) override;

  // Implement anariNewArray3D()
  ANARIArray3D newArray3D(const void *appMemory,
      ANARIMemoryDeleter deleter,
      const void *userdata,
      ANARIDataType,
      uint64_t numItems1,
      uint64_t numItems2,
      uint64_t numItems3) override;

  // Implement anariNewGeometry()
  ANARIGeometry newGeometry(const char *type) override;

  // Implement anariNewMaterial()
  ANARIMaterial newMaterial(const char *material_type) override;

  // Implement anariNewSampler()
  ANARISampler newSampler(const char *type) override;

  // Implement anariNewSurface()
  ANARISurface newSurface() override;

  // Implement anariNewSpatialField()
  ANARISpatialField newSpatialField(const char *type) override;

  // Implement anariNewVolume()
  ANARIVolume newVolume(const char *type) override;

  // Implement anariNewLight()
  ANARILight newLight(const char *type) override;

  // Implement anariGroup()
  ANARIGroup newGroup() override;

  // Implement anariNewInstance()
  ANARIInstance newInstance(const char *type) override;

  // Implement anariNewWorld()
  ANARIWorld newWorld() override;

  // Implement anariNewCamera()
  ANARICamera newCamera(const char *type) override;

  // Implement anariNewRenderer()
  ANARIRenderer newRenderer(const char *type) override;

  // Implement anariNewFrame()
  ANARIFrame newFrame() override;

  // Object + Parameter Lifetime Management ///////////////////////////////////

  // Implement anariSetParameter()
  void setParameter(ANARIObject object,
      const char *name,
      ANARIDataType type,
      const void *mem) override;

  // Implement anariUnsetParameter()
  void unsetParameter(ANARIObject object, const char *name) override;

  // Implement anariUnsetAllParameters()
  void unsetAllParameters(ANARIObject object) override;

  // Implement anariMapParameterArray1D()
  void *mapParameterArray1D(ANARIObject object,
      const char *name,
      ANARIDataType dataType,
      uint64_t numElements1,
      uint64_t *elementStride) override;

  // Implement anariMapParameterArray2D()
  void *mapParameterArray2D(ANARIObject object,
      const char *name,
      ANARIDataType dataType,
      uint64_t numElements1,
      uint64_t numElements2,
      uint64_t *elementStride) override;

  // Implement anariMapParameterArray3D()
  void *mapParameterArray3D(ANARIObject object,
      const char *name,
      ANARIDataType dataType,
      uint64_t numElements1,
      uint64_t numElements2,
      uint64_t numElements3,
      uint64_t *elementStride) override;

  // Implement anariUnmapParameterArray()
  void unmapParameterArray(ANARIObject object, const char *name) override;

  // Implement anariCommitParameters()
  void commitParameters(ANARIObject object) override;

  // Implement anariRelease()
  void release(ANARIObject _obj) override;

  // Implement anariRetain()
  void retain(ANARIObject _obj) override;

  // Implement anariMapArray()
  void *mapArray(ANARIArray) override;

  // Implement anariUnmapArray()
  void unmapArray(ANARIArray) override;

  // Implement anariGetProperty()
  int getProperty(ANARIObject object,
      const char *name,
      ANARIDataType type,
      void *mem,
      uint64_t size,
      ANARIWaitMask mask) override;

  // Object Query Interface ///////////////////////////////////////////////////

  // Implement anariGetObjectSubtypes()
  const char **getObjectSubtypes(ANARIDataType objectType) override;

  // Implement anariGetObjectInfo()
  const void *getObjectInfo(ANARIDataType objectType,
      const char *objectSubtype,
      const char *infoName,
      ANARIDataType infoType) override;

  // Implement anariGetParameterInfo()
  const void *getParameterInfo(ANARIDataType objectType,
      const char *objectSubtype,
      const char *parameterName,
      ANARIDataType parameterType,
      const char *infoName,
      ANARIDataType infoType) override;

  // FrameBuffer Manipulation /////////////////////////////////////////////////

  // Implement anariFrameBufferMap
  const void *frameBufferMap(ANARIFrame fb,
      const char *channel,
      uint32_t *width,
      uint32_t *height,
      ANARIDataType *pixelType) override;

  // Implement anariFrameBufferUnmap
  void frameBufferUnmap(ANARIFrame fb, const char *channel) override;

  // Frame Rendering //////////////////////////////////////////////////////////

  // Implement anariRenderFrame()
  void renderFrame(ANARIFrame) override;

  // Implement anariFrameReady()
  int frameReady(ANARIFrame, ANARIWaitMask) override;

  // Implement anariDiscardFrame()
  void discardFrame(ANARIFrame) override;

  /////////////////////////////////////////////////////////////////////////////
  // Helper/other functions and data members
  /////////////////////////////////////////////////////////////////////////////

  PTCDevice(ANARIStatusCallback defaultCallback, const void *userPtr);
  PTCDevice(ANARILibrary);
  ~PTCDevice() override;

  void initDevice();

 private:
  void deviceCommitParameters();
  int deviceGetProperty(
      const char *name, ANARIDataType type, void *mem, uint64_t size);

  bool isFrameHandle(ANARIObject o);
  void removeFrameWrapper(FrameWrapper *fw);

  template <typename... Args>
  void reportMessage(
      ANARIStatusSeverity, const char *fmt, Args &&...args) const;

  bool m_initialized{false};
  uint32_t m_refCount{1};
  anari::DeviceImpl *m_ptd{nullptr};
  std::vector<FrameWrapper *> m_frameWrappers;

  static int s_numMpiPTCDevices;
  static bool s_mpiInitializedPrivately;
};

// Inlined definitions ////////////////////////////////////////////////////////

template <typename... Args>
inline void PTCDevice::reportMessage(
    ANARIStatusSeverity severity, const char *fmt, Args &&...args) const
{
  auto statusCB = defaultStatusCallback();
  if (!statusCB)
    return;

  const void *statusCBUserPtr = defaultStatusCallbackUserPtr();

  auto msg = helium::string_printf(fmt, std::forward<Args>(args)...);

  statusCB(statusCBUserPtr,
      this_device(),
      this_device(),
      ANARI_OBJECT,
      severity,
      severity <= ANARI_SEVERITY_WARNING ? ANARI_STATUS_NO_ERROR
                                         : ANARI_STATUS_UNKNOWN_ERROR,
      msg.c_str());
}

} // namespace ptc
