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
#include "anari/backend/LibraryImpl.h"
#include "anari_library_ptc_export.h"

namespace ptc {

struct PTCLibrary : public anari::LibraryImpl
{
  PTCLibrary(
      void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr);

  ANARIDevice newDevice(const char *subtype) override;
  const char **getDeviceExtensions(const char *deviceType) override;
};

// Definitions ////////////////////////////////////////////////////////////////

PTCLibrary::PTCLibrary(
    void *lib, ANARIStatusCallback defaultStatusCB, const void *statusCBPtr)
    : anari::LibraryImpl(lib, defaultStatusCB, statusCBPtr)
{}

ANARIDevice PTCLibrary::newDevice(const char * /*subtype*/)
{
  return (ANARIDevice) new PTCDevice(this_library());
}

const char **PTCLibrary::getDeviceExtensions(const char * /*deviceType*/)
{
  return nullptr;
}

} // namespace ptc

// Define library entrypoint //////////////////////////////////////////////////

extern "C" PTC_EXPORT ANARI_DEFINE_LIBRARY_ENTRYPOINT(
    ptc, handle, scb, scbPtr)
{
  return (ANARILibrary) new ptc::PTCLibrary(handle, scb, scbPtr);
}
