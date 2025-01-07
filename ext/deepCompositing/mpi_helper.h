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

#pragma once

#include <mpi.h>

#define MPI_CALL(a) {                                                          \
  int rc = MPI_##a;                                                            \
  if (rc != MPI_SUCCESS) {                                                     \
    static char estring[MPI_MAX_ERROR_STRING];                                 \
    int len;                                                                   \
    MPI_Error_string(rc, estring, &len);                                       \
    throw std::runtime_error("mpi error ..."+std::string(estring,len));        \
  }                                                                            \
}

