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

#include "CPUCompositor.h"
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <mutex>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

namespace ptc {

  template<typename TASK_T>
  inline void parallel_for(size_t nTasks,
                           TASK_T&& taskFunction,
                           size_t blockSize=1)
  {
    if (nTasks == 0) return;
    if (nTasks == 1)
      taskFunction(size_t(0));
    else if (blockSize==1) {
      tbb::parallel_for(size_t(0), nTasks, std::forward<TASK_T>(taskFunction));
    } else {
      const size_t numBlocks = (nTasks+blockSize-1)/blockSize;
      tbb::parallel_for((size_t)0, numBlocks, [&](size_t blockIdx){
        size_t begin = blockIdx*blockSize;
        size_t end   = std::min(begin+blockSize,size_t(nTasks));
        for (size_t i=begin;i<end;i++)
          taskFunction(size_t(i));
      });
    }
  }

  template<typename TASK_T>
  void parallel_for_blocked(size_t begin, size_t end, size_t blockSize,
                            const TASK_T& taskFunction)
  {
    const size_t numTasks = end - begin;
    const size_t numBlocks = (numTasks + blockSize - 1) / blockSize;
    parallel_for(numBlocks, [&](size_t blockID) {
      size_t block_begin = begin + blockID * blockSize;
      taskFunction(block_begin, std::min(block_begin + blockSize, end));
    });
  }
  
  int divRoundUp(int a, int b) { return (a+b-1)/b; }

  struct Fragment {
    inline uint32_t rgba8() const
    {
      uint32_t r = uint8_t(std::min(255.f,color.x * 255.f + .5f));
      uint32_t g = uint8_t(std::min(255.f,color.y * 255.f + .5f));
      uint32_t b = uint8_t(std::min(255.f,color.z * 255.f + .5f));
      uint32_t a = uint8_t(std::min(255.f,alpha * 255.f + .5f));
      return r | (g << 8) | (b << 16) | (a << 24);
    }
    math::float3 color;
    float        alpha;
  };
  
  struct LongFrag {
    static LongFrag make(uint32_t color, float depth)
    { return { (size_t((const uint32_t &)depth) << 32) | color }; };

    inline float depth() const {
      const uint32_t *p = (const uint32_t *)&bits;
      return (const float &)p[1];
    }
    
    inline Fragment color() const
    {
      float r = ((bits >>  0) & 0xff) * (1.f/255.f);
      float g = ((bits >>  8) & 0xff) * (1.f/255.f);
      float b = ((bits >> 16) & 0xff) * (1.f/255.f);
      float a = ((bits >> 24) & 0xff) * (1.f/255.f);
      Fragment ret;
      ret.color.x = r;
      ret.color.y = g;
      ret.color.z = b;
      ret.alpha = a;
      return ret;
    }
    
    uint64_t bits;
  };
  
  CPUCompositor::CPUCompositor(MPI_Comm comm)
  {
    mpi.comm = comm;
    MPI_Comm_rank(comm,&mpi.rank);
    MPI_Comm_size(comm,&mpi.size);
    printf("#cpu compositor starting up, ranks %i/%i\n",mpi.rank,mpi.size);
    p2p.sendOffset.resize(mpi.size);
    p2p.sendCount.resize(mpi.size);
    p2p.recvOffset.resize(mpi.size);

    if (mpi.rank == 0) {
      finalGather.rank0.recvCount.resize(mpi.size);
      finalGather.rank0.recvOffset.resize(mpi.size);
    }
  }

  CPUCompositor::Range CPUCompositor::rangeOfPeer(int peer)
  {
    int numPixels = size.x*size.y;
    int numChunks = divRoundUp(numPixels,(int)chunkSize);
    int chunk_begin = (mpi.rank * numChunks) / mpi.size;
    int chunk_end = ((mpi.rank+1) * numChunks) / mpi.size;
    return Range(chunk_begin*chunkSize,
                 std::min(chunk_end*chunkSize,numPixels));
  }
  
  void CPUCompositor::resize(int sx, int sy)
  {
    assert(sx > 0);
    assert(sy > 0);
    size = { sx,sy };
    
    Range myRange = rangeOfPeer(mpi.rank);
    int recvBufSize = myRange.size() * mpi.size;
    p2p.recvBuf.color.resize(recvBufSize);
    p2p.recvBuf.depth.resize(recvBufSize);
    finalGather.anyRank.sendOffset = 0;
    finalGather.anyRank.sendCount  = myRange.size();
    finalGather.anyRank.sendBuf.color.resize(myRange.size());
    finalGather.anyRank.sendBuf.depth.resize(myRange.size());

    if (mpi.rank == 0) {
      /* compute receive offsets into FINAL frame buffer (Only need
         that on rank 0, but anyway...) */
      for (int peer=0;peer<mpi.size;peer++) {
        Range peerRange = rangeOfPeer(peer);
        finalGather.rank0.recvOffset[peer] = peerRange.begin;
        finalGather.rank0.recvCount[peer]  = peerRange.size();
      }
    }
      

    { /* compute receive offsets into peer-to-peer exchange state */
      int recvOffset = 0;
      int sendOffset = 0;
      for (int peer=0;peer<mpi.size;peer++) {
        Range peerRange = rangeOfPeer(peer);
        
        p2p.sendOffset[peer] = sendOffset;
        p2p.recvOffset[peer] = recvOffset;
        p2p.sendCount[peer] = peerRange.size();
        p2p.recvCount[peer] = myRange.size();

        recvOffset += p2p.recvOffset[peer];
        sendOffset += p2p.sendOffset[peer];
      }
    }
  }
  
  /*! out_color and out_depth should be null on all ranks > 0 */
  void CPUCompositor::run(uint32_t *out_color,
                          float    *out_depth,
                          const uint32_t *in_color,
                          const float    *in_depth)
  {
    // ------------------------------------------------------------------
    // peer-to-peer stage
    // ------------------------------------------------------------------
    MPI_Alltoallv(in_color,
                       p2p.sendCount.data(),
                       p2p.sendOffset.data(),
                       MPI_INT,
                       p2p.recvBuf.color.data(),
                       p2p.recvCount.data(),
                       p2p.recvOffset.data(),
                       MPI_INT,
                  mpi.comm);
    MPI_Alltoallv(in_depth,
                  p2p.sendCount.data(),
                  p2p.sendOffset.data(),
                  MPI_FLOAT,
                  p2p.recvBuf.depth.data(),
                  p2p.recvCount.data(),
                  p2p.recvOffset.data(),
                  MPI_FLOAT,
                  mpi.comm);

    // // ------------------------------------------------------------------
    // // local compositing stage
    // // ------------------------------------------------------------------
    Range myRange = rangeOfPeer(mpi.rank);
    parallel_for_blocked
      (0,(size_t)myRange.size(),blockSize,[&](size_t begin,size_t end) {
        // size_t begin = blockID*blockSize;
        // size_t end = std::min(begin+blockSize,myRange.end);
        std::vector<LongFrag> fragments(mpi.size);
        const uint32_t *colors = p2p.recvBuf.color.data();
        const float    *depths = p2p.recvBuf.depth.data();
        size_t rankStep = myRange.size();
        for (size_t it=begin;it<end;it++) {
          for (size_t r=0;r<(size_t)mpi.size;r++) {
            size_t ofs = it + r*rankStep;
            fragments[r] = LongFrag::make(colors[ofs],depths[ofs]);
          }
          std::sort(&fragments.begin()->bits,
                    &fragments.end()->bits);
          Fragment composited = fragments.back().color();
          for (int64_t r=(int64_t)mpi.size-2;r>=0;--r) {
            Fragment thisFrag = fragments[r].color();
            composited.color
              = thisFrag.alpha * thisFrag.color
              + (1.f-thisFrag.alpha) * composited.color;
            composited.alpha
              = thisFrag.alpha
              + (1.f-thisFrag.alpha) * composited.alpha;
          }
          float    final_depth = fragments[0].depth();
          uint32_t final_color = composited.rgba8();

          finalGather.anyRank.sendBuf.color[it] = final_color;
          finalGather.anyRank.sendBuf.depth[it] = final_depth;
        }
      });

    // ------------------------------------------------------------------
    // final gather stage
    // ------------------------------------------------------------------
    if (mpi.rank == 0) {
      std::vector<MPI_Request> color_requests(mpi.size);
      std::vector<MPI_Request> depth_requests(mpi.size);
      for (int r=0;r<mpi.size;r++) {
        MPI_Irecv(out_color+finalGather.rank0.recvOffset[r],
                  finalGather.rank0.recvCount[r],
                  MPI_INT,r,0,mpi.comm,&color_requests[r]);
        MPI_Irecv(out_depth+finalGather.rank0.recvOffset[r],
                  finalGather.rank0.recvCount[r],
                  MPI_FLOAT,r,0,mpi.comm,&depth_requests[r]);
      }
      MPI_Waitall(color_requests.size(),
                  color_requests.data(),
                  MPI_STATUSES_IGNORE);
      MPI_Waitall(depth_requests.size(),
                  depth_requests.data(),
                  MPI_STATUSES_IGNORE);
    } else {
      Range myRange = rangeOfPeer(mpi.rank);
      MPI_Request colorRequest;
      MPI_Request depthRequest;
      MPI_Isend(in_color+finalGather.anyRank.sendOffset,
                finalGather.anyRank.sendCount,
                MPI_INT,0,0,mpi.comm,&colorRequest);
      MPI_Isend(in_depth+finalGather.anyRank.sendOffset,
                finalGather.anyRank.sendCount,
                MPI_FLOAT,0,0,mpi.comm,&depthRequest);
      MPI_Wait(&colorRequest,MPI_STATUS_IGNORE);
      MPI_Wait(&depthRequest,MPI_STATUS_IGNORE);
    }
              
  }
  
}
