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

// #include "owl/common/math/vec.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <iostream>

/* in this strategy we find the 'weakest' fragment (as in lowest
   opeacity), and merge this with the following fragment */
// #define OVERFLOW_STRATEGY_MERGE_WEAKEST_FRAGMENT 1

#define DEFAULT_MAX_FRAGS_PER_PIXEL 3

namespace dc {

  class TwoPassInterface;
  class FixedFragsPerPixel;

#define SMALL_FRAGMENTS 1

  inline int divRoundUp(int a, int b) { return (a+b-1)/b; }
  inline __device__ float4 operator*(float f, float4 v)
  { return make_float4(f*v.x,f*v.y,f*v.z,f*v.w); }
  inline __device__ float3 operator*(float f, float3 v)
  { return make_float3(f*v.x,f*v.y,f*v.z); }

  inline __device__ float4 operator+(float4 a, float4 b)
  { return make_float4(a.x+b.x,
                       a.y+b.y,
                       a.z+b.z,
                       a.w+b.w); }

  inline __device__ float3 operator*(float3 a, float3 b)
  { return make_float3(a.x*b.x,
                       a.y*b.y,
                       a.z*b.z); }
  inline __device__ float3 operator+(float3 a, float3 b)
  { return make_float3(a.x+b.x,
                       a.y+b.y,
                       a.z+b.z); }

#ifndef __CUDA_ARCH__
  using std::min;
  using std::max;
#endif

#if SMALL_FRAGMENTS


  struct Fragment {
    inline// __device__
    Fragment() = default;
    inline __device__ Fragment(float z, const float3 &color, const float alpha=1.f)
      : r(encode8(color.x)),
        g(encode8(color.y)),
        b(encode8(color.z)),
        a(encode8(alpha)),
        z(min(z,1e19f))
    {}
    inline __device__ Fragment(float z, const float4 &color)
      : r(encode8(color.x)),
        g(encode8(color.y)),
        b(encode8(color.z)),
        a(encode8(color.w)),
        z(min(z,1e19f))
    {}

    inline __device__ __host__ float4 getRGBA() const {
      return make_float4(r*(1.f/255.f),
                         g*(1.f/255.f),
                         b*(1.f/255.f),
                         a*(1.f/255.f)
                         );
    }
    inline __device__ __host__ float getAlpha() const {
      return a*(1.f/255.f);
    }

    static inline __device__ uint8_t encode8(float f)
    { return uint8_t(max(0.f,min(255.f,255.f*f+.5f))); }

    float z;
    uint8_t r,g,b;
  private:
    uint8_t a;
  };
#else
  struct Fragment {
    inline// __device__
    Fragment() = default;
    inline __device__ Fragment(float z, const float3 &color, const float alpha=1.f)
      : r(color.x),g(color.y),b(color.z),a(alpha),z(min(z,1e19f))
    {}
    inline __device__ Fragment(float z, const float4 &color)
      : r(color.x),g(color.y),b(color.z),a(color.w),z(min(z,1e19f))
    {}
    inline __device__ __host__ float4 getRGBA() const { return make_float4(r,g,b,a); }
    inline __device__ __host__ float getAlpha() const { return a; }

    float r,g,b,a;
    float z;
  };
#endif

  /*! device-side interface to a 'deep frame buffer'; this is the
      interface that a device-side renderer will use to write
      fragments */
  struct DeviceInterface
  {
    /*! write given fragment into given pixel's fragment list
        (possibly discarding fragment or replacing and existing one if
        already full), and return length of list after fragment is
        written */
    inline __device__
    uint32_t write(const int pixelID_x,
                   const int pixelID_y,
                   const Fragment &fragment,
                   bool dbg=false) const;

    inline __device__
    int2 getSize() const { return fbSize; }

    /*! array of all fragments, using N fragments per each pixel */
    Fragment *fragments;
    /*! exactly one counter per pixel */
    uint32_t *counters;

    int2 fbSize;
    int maxFragsPerPixel;
  };

  /*! for debugging and benchmarking - compacted list of fragments
      from a previous render pass, can be used to save and replay */
  struct SavedFrags {
    /*! for debugging - load output from prev 'save' back in */
    static SavedFrags load(const std::string &fileName);

    int2 fbSize { 0,0 };
    /* one int per pixel, pointing to _end_ of fragemnt list for this
       pixel - this is cuda managed mem */
    uint32_t *counters { 0 };

    /*! list of all fragments, compacted - this is cuda managed mem */
    Fragment *fragments { 0 };
  };

  /*! the compositor itself, that only operates on a) a compact array
      of fragments, b) a per-pixel array of fragment counts, and c) a
      per-pixel array of fragment list offsets (into the compact
      fragment array) on each rank. */
  struct Compositor
  {
    Compositor(int maxFragsPerPixel=DEFAULT_MAX_FRAGS_PER_PIXEL,
               MPI_Comm comm = MPI_COMM_WORLD);

    /*! prepare rendering the next frame - MUST be called exactly once
        before rendering the frame */
    DeviceInterface prepare();

    /*! run the actual mpi+cuda compositing stage, using the 'fbSize',
        'pixelCounters' and 'pixelOffsets' values that the Interface
        has set up. Composited values will get written to
        'whereToWriteFinalPixels' on rank 0 (where this value must not
        be null); all other ranks should pass null here */
    void finish(uint32_t *whereToWriteFinalPixels);

    /*! resize the (device) frame buffer to given size */
    void resize(const int2 &size);

    /*! resize the (device) frame buffer to given size */
    void resize(const int size_x, const int size_y) { resize({size_x,size_y}); }


    /*! pick one of the local nodes' GPUs, base on how many other
      ranks are runnong on that same node. returns the GPU picked
      for this rank. ranks are relative to the 'comm' communicator
      passed to the constructor */
    int  affinitizeGPU();

    /*! enable peer access between all gpus ... not sure we actulaly
      need this?!? */
    // void enablePeerAccess();

  private:
    /*! for debugging - save current state to disk */
    void save(const std::string &fileName);

  public:
    /*! used to select which GPUs to run on (in case of multiple
      GPUs in the node and multiple rankson that same node. If -1,
      we'll not explicit call cudaSetGPU at all, and leave it to the
      user/app to select this; if instead user calls
      affinitizeGPU(), then we'll pick a GPU based on which ranks
      are runnong on which GPU, and will always call cudaSetGPU()
      (with that GPU, obviously) before doing any GPU calls) */
    int       affinitizedGPU { -1 };

    DeviceInterface deviceData = {};

    int2      fbSize          { 0,0 };

    /*! one int per pixel, telling where the given pixel's fragment
        list (in the localFramgements[] array) _ends_ (the beginning
        of the list is given by either the end of the previous pixel's
        list (if pixelID > 0), or 0 (for pixel 0); this is the same
        array as DeviceInterface::offsets; the values are computes as
        a inclusive scan after the frame renderered */
    uint32_t *fullIntCounters    { nullptr };

    /*! compacted counters: during rendeirng counters are stored as
        32-bit uints (to re-use the array used for offstes, and to
        allow atomic operaiton on them); but transmission in this form
        would be very wasteful, so we always combine multiple pixels
        to a single 8-bit value */
    uint8_t  *lowBitCounters { nullptr };

    /*! not yet compacted fragment lists */
    Fragment *fixedSizeFragmentLists { nullptr };
    /*! compact array of all pixels' fragment list; this list first
        stores all the framgemnets from pixel 0, then those from pixel
        1, etc.; its size is dependent on how many pixels there are in
        the entire frame buffer; allocating this porperly is the job
        of the respectvie interface */
    Fragment *compactedFragmentLists  { nullptr };

    const uint32_t maxFragsPerPixel;
    const uint32_t numCountersPerByte;

    inline uint32_t pixelBegin(uint32_t nodeID) const;
    inline uint32_t pixelEnd(uint32_t nodeID) const
    { return pixelBegin(nodeID+1); }

    inline uint32_t computeNumPixelsOnThisRank() const
    { return pixelEnd(rank) - pixelBegin(rank); }

    inline uint32_t numRanks() const { return size; }
    //    inline uint32_t numPixels() const { return fbSize.x*fbSize.y; }

    // number of pixels in a logical screen padded to be a multile of countersPerByte pixels
    uint32_t numPixelsPadded = 0;
    // the actual number of pixels on the user side, NOT padded
    uint32_t numPixelsOrg = 0;

    /*! @{ MPI stuff */
    MPI_Comm  comm            { MPI_COMM_WORLD };
    int       rank            { -1 };
    int       size            { -1 };
    /*! @} */

    /*! base of a filename in which every rank will save the rendered
        fragments *before* compositing, but *after* computing the
        compacting the per-pixel fragment lists - this is to allow
        easy "replays" in dcBench tool */
    static std::string dumpFileName;
  };

  /*! write given fragment into given pixel's fragment list - in this
      implementation we always insert in a sorted way; ie, the list is
      always kept sorted during insertion of any new fragments, with
      existing fragsments 'behin' the newly inserted fragments being
      pushed one positoin to the end. If the list is already full when
      a new fragment gets written the most distant fragment gets
      thrown away. Note an alternative strategy would be to throw away
      the fragment with lowst opacity, but for now we do only this. */
  inline __device__
  uint32_t DeviceInterface::write(const int pixelID_x,
                                  const int pixelID_y,
                                  const Fragment &fragment,
                                  bool dbg) const
  {
    int idx = pixelID_x + fbSize.x*pixelID_y;
    Fragment *frags = fragments+idx*maxFragsPerPixel;
    int listLength = counters[idx];
    if (listLength == maxFragsPerPixel) {
#if OVERFLOW_STRATEGY_MERGE_WEAKEST_FRAGMENT
      /* in this strategy we find the 'weakest' fragment (as in lowest
         opeacity), and merge this with the following fragment */
      int lowest_pos = 0;
      float lowest_val = frags[0].getAlpha();
      for (int i=1;i<listLength;i++)
        if (frags[i].getAlpha() < lowest_val) {
          lowest_val = frags[i].getAlpha();
          lowest_pos = i;
        }
      if (lowest_pos == 0) lowest_pos = 1;
      float4 c0 = frags[lowest_pos-1].getRGBA();
      float4 c1 = frags[lowest_pos].getRGBA();
      c0 = c0 + (1.f-c0.w)*c1;
      frags[lowest_pos-1] = Fragment(frags[lowest_pos-1].z,c0);
      for (int i=lowest_pos+1;i<listLength;i++)
        frags[i-1] = frags[i];
      --listLength;
#else
      // list is already full - either discard this, or last in list
      if (float(fragment.z) >= float(frags[listLength-1].z))
        // already behind last element, just throw away
        return listLength;
      else
        // throw last elemnet in list away by reducing list length-
        // this alllows for inserting this one as new one
        --listLength;
#endif
    }
    int insertPos = listLength;
    while (insertPos > 0 && float(frags[insertPos-1].z) > float(fragment.z)) {
      frags[insertPos] = frags[insertPos-1];
      --insertPos;
    }
    frags[insertPos] = fragment;
    listLength++;
    counters[idx] = listLength;
    if (dbg) printf("write: list length %i\n",listLength);
    return listLength;
  }

  inline uint32_t nextMultipleOf(uint32_t i, uint32_t base)
  { return divRoundUp(i,base)*base; }

  inline uint32_t Compositor::pixelBegin(uint32_t nodeID) const
  {
    return nextMultipleOf((nodeID*numPixelsPadded)/numRanks(),
                          numCountersPerByte);
  }


} // ::dc
