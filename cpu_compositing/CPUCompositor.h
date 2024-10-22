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

#include <mpi.h>
#include <functional>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <map>
#include <anari/anari_cpp/ext/linalg.h>
#include <iostream>

namespace ptc {
  using namespace anari;
  
#ifndef PRINT
# define PRINT(var) std::cout << #var << "=" << var << std::endl;
#ifdef __WIN32__
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__ << std::endl;
#else
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__ << std::endl;
#endif
#endif

  struct CPUCompositor
  {
    CPUCompositor(MPI_Comm comm);

    void resize(int sx, int sy);
    /*! out_color and out_depth should be null on all ranks > 0 */
    void run(uint32_t *out_color,
             float    *out_depth,
             const uint32_t *in_color,
             const float    *in_depth);
    struct Range {
      inline Range() = default;
      inline Range(int begin, int end) : begin(begin), end(end) {}
      inline int size() const { return end-begin; }
      int begin=0, end=0;
    };
  private:
    math::int2 size { 0,0 };


    Range rangeOfPeer(int peer);
    
    struct {
      struct {
        std::vector<uint32_t> color;
        std::vector<float>    depth;
      } recvBuf;
      Range myRange;
      std::vector<int> recvOffset;
      std::vector<int> recvCount;
      std::vector<int> sendOffset;
      std::vector<int> sendCount;
    } p2p;
    struct {
      struct {
        std::vector<int> recvOffset;
        std::vector<int> recvCount;
      } rank0;
      struct {
        int sendOffset;
        int sendCount;
        struct {
          std::vector<uint32_t> color;
          std::vector<float>    depth;
        } sendBuf;
      } anyRank;
    } finalGather;
    
    struct {
      MPI_Comm comm;
      int size;
      int rank;
    } mpi;
    enum { chunkSize = 64 };
    enum { blockSize = 16 * chunkSize };
  };
  
}
