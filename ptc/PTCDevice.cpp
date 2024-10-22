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

#include "PTCDevice.h"
// helium
#include <helium/BaseGlobalDeviceState.h>
#include <helium/BaseObject.h>
// std
#include <algorithm>
#include <memory>
#if PTC_CUDA
# include "CUDAFrameWrapper.h"
#else
# include "CPUFrameWrapper.h"
#endif

namespace ptc {

int PTCDevice::s_numMpiPTCDevices = 0;
bool PTCDevice::s_mpiInitializedPrivately = false;

// Helper functions ///////////////////////////////////////////////////////////

template <typename T, typename U>
static bool sameAddress(const T *p1, const U *p2)
{
  return (const void *)p1 == (const void *)p2;
}

static FrameWrapper *asFrameWrapper(ANARIObject o)
{
  return reinterpret_cast<FrameWrapper *>(o);
}

// API Objects ////////////////////////////////////////////////////////////////

ANARIArray1D PTCDevice::newArray1D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems)
{
  initDevice();
  return m_ptd->newArray1D(appMemory, deleter, userData, type, numItems);
}

ANARIArray2D PTCDevice::newArray2D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2)
{
  initDevice();
  return m_ptd->newArray2D(
      appMemory, deleter, userData, type, numItems1, numItems2);
}

ANARIArray3D PTCDevice::newArray3D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2,
    uint64_t numItems3)
{
  initDevice();
  return m_ptd->newArray3D(
      appMemory, deleter, userData, type, numItems1, numItems2, numItems3);
}

ANARICamera PTCDevice::newCamera(const char *subtype)
{
  initDevice();
  return m_ptd->newCamera(subtype);
}

ANARIFrame PTCDevice::newFrame()
{
  initDevice();
#if PTC_CUDA
  typedef CUDAFrameWrapper FrameWrapperT;
#else
  typedef CPUFrameWrapper FrameWrapperT;
#endif
  auto *fw = new FrameWrapperT((ANARIDevice)m_ptd,
      m_ptd->newFrame(),
      [=](const void *f) { this->removeFrameWrapper((FrameWrapper *)f); });
  m_frameWrappers.push_back(fw);
  return (ANARIFrame)fw;
}

ANARIGeometry PTCDevice::newGeometry(const char *subtype)
{
  initDevice();
  return m_ptd->newGeometry(subtype);
}

ANARIGroup PTCDevice::newGroup()
{
  initDevice();
  return m_ptd->newGroup();
}

ANARIInstance PTCDevice::newInstance(const char *subtype)
{
  initDevice();
  return m_ptd->newInstance(subtype);
}

ANARILight PTCDevice::newLight(const char *subtype)
{
  initDevice();
  return m_ptd->newLight(subtype);
}

ANARIMaterial PTCDevice::newMaterial(const char *subtype)
{
  initDevice();
  return m_ptd->newMaterial(subtype);
}

ANARIRenderer PTCDevice::newRenderer(const char *subtype)
{
  initDevice();
  return m_ptd->newRenderer(subtype);
}

ANARISampler PTCDevice::newSampler(const char *subtype)
{
  initDevice();
  return m_ptd->newSampler(subtype);
}

ANARISpatialField PTCDevice::newSpatialField(const char *subtype)
{
  initDevice();
  return m_ptd->newSpatialField(subtype);
}

ANARISurface PTCDevice::newSurface()
{
  initDevice();
  return m_ptd->newSurface();
}

ANARIVolume PTCDevice::newVolume(const char *subtype)
{
  initDevice();
  return m_ptd->newVolume(subtype);
}

ANARIWorld PTCDevice::newWorld()
{
  initDevice();
  return m_ptd->newWorld();
}

// Object + Parameter Lifetime Management /////////////////////////////////////

void PTCDevice::setParameter(
    ANARIObject object, const char *name, ANARIDataType type, const void *mem)
{
  if (handleIsDevice(object))
    this->setParam(name, type, mem);
  else if (isFrameHandle(object))
    asFrameWrapper(object)->setParameter(name, type, mem);
  else
    m_ptd->setParameter(object, name, type, mem);
}

void PTCDevice::unsetParameter(ANARIObject object, const char *name)
{
  if (handleIsDevice(object))
    this->removeParam(name);
  else if (isFrameHandle(object))
    asFrameWrapper(object)->unsetParameter(name);
  else
    m_ptd->unsetParameter(object, name);
}

void PTCDevice::unsetAllParameters(ANARIObject object)
{
  if (handleIsDevice(object))
    this->removeAllParams();
  else if (isFrameHandle(object))
    asFrameWrapper(object)->unsetAllParameters();
  else
    m_ptd->unsetAllParameters(object);
}

void *PTCDevice::mapParameterArray1D(ANARIObject object,
    const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t *elementStride)
{
  if (isFrameHandle(object)) {
    return asFrameWrapper(object)->mapParameterArray1D(
        name, dataType, numElements1, elementStride);
  } else {
    return m_ptd->mapParameterArray1D(
        object, name, dataType, numElements1, elementStride);
  }
}

void *PTCDevice::mapParameterArray2D(ANARIObject object,
    const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t *elementStride)
{
  if (isFrameHandle(object)) {
    return asFrameWrapper(object)->mapParameterArray2D(
        name, dataType, numElements1, numElements2, elementStride);
  } else {
    return m_ptd->mapParameterArray2D(
        object, name, dataType, numElements1, numElements2, elementStride);
  }
}

void *PTCDevice::mapParameterArray3D(ANARIObject object,
    const char *name,
    ANARIDataType dataType,
    uint64_t numElements1,
    uint64_t numElements2,
    uint64_t numElements3,
    uint64_t *elementStride)
{
  if (isFrameHandle(object)) {
    return asFrameWrapper(object)->mapParameterArray3D(name,
        dataType,
        numElements1,
        numElements2,
        numElements3,
        elementStride);
  } else {
    return m_ptd->mapParameterArray3D(object,
        name,
        dataType,
        numElements1,
        numElements2,
        numElements3,
        elementStride);
  }
}

void PTCDevice::unmapParameterArray(ANARIObject object, const char *name)
{
  if (isFrameHandle(object))
    asFrameWrapper(object)->unmapParameterArray(name);
  else
    m_ptd->unmapParameterArray(object, name);
}

void PTCDevice::commitParameters(ANARIObject object)
{
  if (handleIsDevice(object))
    deviceCommitParameters();
  else if (isFrameHandle(object))
    asFrameWrapper(object)->commitParameters();
  else
    m_ptd->commitParameters(object);
}

void PTCDevice::release(ANARIObject o)
{
  if (o == nullptr)
    return;

  if (handleIsDevice(o)) {
    if (--m_refCount == 0)
      delete this;
    return;
  } else if (isFrameHandle(o))
    asFrameWrapper(o)->release();
  else
    m_ptd->release(o);
}

void PTCDevice::retain(ANARIObject o)
{
  if (handleIsDevice(o))
    m_refCount++;
  else if (isFrameHandle(o))
    asFrameWrapper(o)->retain();
  else
    m_ptd->retain(o);
}

void *PTCDevice::mapArray(ANARIArray a)
{
  return m_ptd->mapArray(a);
}

void PTCDevice::unmapArray(ANARIArray a)
{
  m_ptd->unmapArray(a);
}

int PTCDevice::getProperty(ANARIObject object,
    const char *name,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    ANARIWaitMask mask)
{
  if (handleIsDevice(object))
    return deviceGetProperty(name, type, mem, size);
  else if (isFrameHandle(object))
    return asFrameWrapper(object)->getProperty(name, type, mem, size, mask);
  else
    return m_ptd->getProperty(object, name, type, mem, size, mask);
}

// Query functions ////////////////////////////////////////////////////////////

const char **PTCDevice::getObjectSubtypes(ANARIDataType objectType)
{
  initDevice();
  return m_ptd->getObjectSubtypes(objectType);
}

const void *PTCDevice::getObjectInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *infoName,
    ANARIDataType infoType)
{
  initDevice();
  return m_ptd->getObjectInfo(objectType, objectSubtype, infoName, infoType);
}

const void *PTCDevice::getParameterInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *parameterName,
    ANARIDataType parameterType,
    const char *infoName,
    ANARIDataType infoType)
{
  initDevice();
  return m_ptd->getParameterInfo(objectType,
      objectSubtype,
      parameterName,
      parameterType,
      infoName,
      infoType);
}

// FrameBuffer Manipulation ///////////////////////////////////////////////////

const void *PTCDevice::frameBufferMap(ANARIFrame f,
    const char *channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  return asFrameWrapper(f)->frameBufferMap(channel, width, height, pixelType);
}

void PTCDevice::frameBufferUnmap(ANARIFrame f, const char *channel)
{
  asFrameWrapper(f)->frameBufferUnmap(channel);
}

// Frame Rendering ////////////////////////////////////////////////////////////

void PTCDevice::renderFrame(ANARIFrame f)
{
  asFrameWrapper(f)->renderFrame();
}

int PTCDevice::frameReady(ANARIFrame f, ANARIWaitMask m)
{
  return asFrameWrapper(f)->frameReady(m);
}

void PTCDevice::discardFrame(ANARIFrame f)
{
  asFrameWrapper(f)->discardFrame();
}

// Other PTCDevice definitions ////////////////////////////////////////////////

PTCDevice::PTCDevice(ANARIStatusCallback defaultCallback, const void *userPtr)
{
  anari::DeviceImpl::m_defaultStatusCB = defaultCallback;
  anari::DeviceImpl::m_defaultStatusCBUserPtr = userPtr;
}

PTCDevice::PTCDevice(ANARILibrary l) : DeviceImpl(l) {}

PTCDevice::~PTCDevice()
{
  reportMessage(ANARI_SEVERITY_DEBUG, "destroying PTC device (%p)", this);

  if (m_ptd)
    m_ptd->release(m_ptd->this_device());

  if (s_mpiInitializedPrivately)
    s_numMpiPTCDevices--;
  if (s_numMpiPTCDevices == 0 && s_mpiInitializedPrivately) {
    reportMessage(ANARI_SEVERITY_DEBUG, "finalizing MPI");
    MPI_Finalize();
  }
}

void PTCDevice::initDevice()
{
  if (m_initialized)
    return;

  reportMessage(ANARI_SEVERITY_DEBUG, "initializing PTC device (%p)", this);

  ANARILibrary lib = nullptr;
  const char *libName = getenv("PTC_ANARI_LIBRARY");
  if (libName) {
    lib = anariLoadLibrary(
        libName, defaultStatusCallback(), defaultStatusCallbackUserPtr());
  }

  if (!lib) {
    reportMessage(ANARI_SEVERITY_WARNING, "falling back to 'helide'");
    lib = anariLoadLibrary(
        "helide", defaultStatusCallback(), defaultStatusCallbackUserPtr());
  }

  if (!lib) {
    reportMessage(ANARI_SEVERITY_FATAL_ERROR, "failed to load passthrough lib");
    return;
  }

  m_ptd = (anari::DeviceImpl *)anariNewDevice(lib, "default");

  if (!m_ptd)
    reportMessage(ANARI_SEVERITY_FATAL_ERROR, "failed to create device");

  int mpiInitialized = 0;
  MPI_Initialized(&mpiInitialized);
  if (s_numMpiPTCDevices == 0 && !mpiInitialized) {
    reportMessage(ANARI_SEVERITY_DEBUG, "initializing MPI");
    MPI_Init(nullptr, nullptr);
    s_mpiInitializedPrivately = true;
    s_numMpiPTCDevices++;
  } else {
    reportMessage(ANARI_SEVERITY_DEBUG, "MPI already initialized");
  }

  m_initialized = true;
}

void PTCDevice::deviceCommitParameters()
{
  // TODO
}

int PTCDevice::deviceGetProperty(
    const char *name, ANARIDataType type, void *mem, uint64_t size)
{
  return 0;
}

bool PTCDevice::isFrameHandle(ANARIObject o)
{
  for (auto *f : m_frameWrappers) {
    if (sameAddress(f, o))
      return true;
  }
  return false;
}

void PTCDevice::removeFrameWrapper(FrameWrapper *fw)
{
  m_frameWrappers.erase(std::remove_if(m_frameWrappers.begin(),
                            m_frameWrappers.end(),
                            [&](auto *f) { return sameAddress(f, fw); }),
      m_frameWrappers.end());
}

} // namespace ptc
