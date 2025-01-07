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

#include "deepCompositing.h"
#include "cuda_helper.h"
#include "mpi_helper.h"
// #include <thrust/scan.h>
// #include <thrust/execution_policy.h>
#include <vector>
#include <sstream>
#include <fstream>
// #include <thrust/device_malloc_allocator.h>
// #include <thrust/device_ptr.h>
#include <cub/cub.cuh>  
    
#define PRINT_STATS 1

namespace dc {

  std::string Compositor::dumpFileName = "";

  struct ProfPrinter
  {
    static int rank;

    ProfPrinter(const char *desc) : desc(desc) {}
    
    void enter() {
      // t_enter = getCurrentTime();
    }
    void leave() {
      // numCalls++; if (numCalls > 0) t_sum += (getCurrentTime()-t_enter);
    }

    ~ProfPrinter()
    {
      // printf("prof(%i) '%s': %ims\n",
      //        rank,desc,int(t_sum/numCalls*1000));
    }
    
    double t_enter;
    double t_sum = 0;
    int numCalls = 0;//-1;
    const char *desc;
  };

  template <typename T>
  inline T download(const T *d_mem, uint32_t ID)
  {
    T t;
    CUDA_CALL(Memcpy(&t,d_mem+ID,sizeof(T),cudaMemcpyDefault));
    return t;
  }
  
  template <typename T>
  void download(std::vector<T> &host, const T *dev)
  {
    CUDA_CALL(Memcpy(host.data(),dev,host.size()*sizeof(T),cudaMemcpyDefault));
  }
  

  int ProfPrinter::rank = -1;

  ProfPrinter prof_all                  ("all together                   ");
  ProfPrinter prof_finalAssemble        ("final master frame assembly    ");
  ProfPrinter prof_cudaCompositing      ("cuda compositing kernel        ");
  ProfPrinter prof_cudaCompactFrags_scan("cuda fragment compaction (scan)");
  ProfPrinter prof_cudaCompactFrags_copy("cuda fragment compaction (copy)");
  ProfPrinter prof_cudaCompactCounters  ("cuda encode counters           ");
  ProfPrinter prof_exchangeCounters     ("exchange counters              ");
  ProfPrinter prof_decodeCounters       ("cuda decode counters           ");
  ProfPrinter prof_computeOffsets       ("compute frag recv offsets      ");
  ProfPrinter prof_exchangeFrags        ("exchange frags                 ");
  
  inline bool operator==(const int2 &a, const int2 &b)
  { return a.x == b.x && a.y == b.y; }
                         
  /*! resize the (device) frame buffer to given size */
  void Compositor::resize(const int2 &newSize)
  {
    if (fbSize == newSize) /* already same size - just skip */ return;
    
    fbSize = newSize;
    deviceData.fbSize = newSize;
    
    int oldDevice = -1;
    if (affinitizedGPU >= 0) {
      CUDA_CALL(GetDevice(&oldDevice));
      CUDA_CALL(SetDevice(affinitizedGPU));
    }

    if (compactedFragmentLists) CUDA_CALL(Free(compactedFragmentLists));
    if (fixedSizeFragmentLists) CUDA_CALL(Free(fixedSizeFragmentLists));
    if (lowBitCounters) CUDA_CALL(Free(lowBitCounters));
    if (fullIntCounters) CUDA_CALL(Free(fullIntCounters));

    this->numPixelsOrg = fbSize.x*fbSize.y;
    this->numPixelsPadded = nextMultipleOf(numPixelsOrg,numCountersPerByte);
    uint32_t compactCounterBytes = numPixelsPadded / numCountersPerByte;

    /* we use the counters[] array for three distinct purposes:

       a) to store the per-pixel counter during rendireng - that's one
       int per pixel, rounded up to how many pixels we have per
       bit-compressed couter

       b) to store the prefix sum offsets during fragement list
       compaction - again one int per pixel

       c) to store all the un-bit-compresed counters after receiving
       all of this ranks' compressed counters - that's
       numRanks*numPixelsOnThisRank ints

       d) the same for offsets into the fragments arrays - again
       numRanks*numPixelsOnThisRank ints
    */
    int numIntsForCountersArray
      = std::max(numPixelsPadded,
                 numRanks()*computeNumPixelsOnThisRank());
    CUDA_CALL(Malloc(&fullIntCounters,numIntsForCountersArray*sizeof(uint32_t)));
    // CUDA_CALL(Malloc(&fullIntCounters,
    //                  nextMultipleOf(numIntsForCountersArray*sizeof(uint32_t),4)));
    // PRINT((int*)fullIntCounters);
    // PRINT(numIntsForCountersArray*sizeof(uint32_t));
    // PRINT(numIntsForCountersArray);
    CUDA_CALL(Memset(fullIntCounters,0,numIntsForCountersArray*sizeof(uint32_t)));
    CUDA_CALL(Malloc(&lowBitCounters,
                     compactCounterBytes*sizeof(uint8_t)));

    /*! same for fragments arrays: we use them for two purposes - once
      for sotring our own fragments during reindering, and then
      again, for receiving compact fragment lists - has to be big
      enough for either */
    int maxNumFragments
      = std::max(/* how many we could possibly produce ourselves: */
                 numPixelsPadded*maxFragsPerPixel,
                 /* how many we could possibly receive: */
                 numRanks()*computeNumPixelsOnThisRank()*maxFragsPerPixel);

    size_t fragArraySize = maxNumFragments*sizeof(Fragment);
    // PRINT(fragArraySize);
    CUDA_CALL(Malloc(&fixedSizeFragmentLists,
                     fragArraySize));
    CUDA_CALL(Malloc(&compactedFragmentLists,
                     fragArraySize));

    if (oldDevice >= 0) {
      CUDA_CALL(SetDevice(oldDevice));
    }
  }

  inline __device__ uint32_t make_8bit(const float f)
  {
    return min(255,max(0,int(f*256.f)));
  }

  inline __device__ uint32_t make_rgba(const float3 color)
  {
    return
      (make_8bit(color.x) << 0) +
      (make_8bit(color.y) << 8) +
      (make_8bit(color.z) << 16) +
      (0xffU << 24);
  }
  inline __device__ uint32_t make_rgba(const float4 color)
  {
    return
      (make_8bit(color.x) << 0) +
      (make_8bit(color.y) << 8) +
      (make_8bit(color.z) << 16) +
      (make_8bit(color.w) << 24);
  }


#define LOG(a) std::cout << a << std::endl;
    
  // void Compositor::enablePeerAccess()
  // {
  //   int totalNumDevices = 0;
  //   cudaGetDeviceCount(&totalNumDevices);

  //   LOG("#dc: enabling peer access ('.'=self, '+'=can access other device)");
  //   int restoreActiveDevice = -1;
  //   cudaGetDevice(&restoreActiveDevice);
  //   int deviceCount = totalNumDevices;//devices.size();
  //   LOG("#dc: found " << deviceCount << " CUDA capable devices");
  //   LOG("#dc: enabling peer access:");
  //   for (int i=0;i<deviceCount;i++) {
  //     std::stringstream ss;
  //     ss << "#dc:  - device #" << i << " : ";
  //     int cuda_i = i;//devices[i]->getCudaDeviceID();
  //     for (int j=0;j<deviceCount;j++) {
  //       if (j == i) {
  //         ss << " o"; 
  //       } else {
  //         int cuda_j = j;//devices[j]->getCudaDeviceID();
  //         int canAccessPeer = 0;
  //         cudaError_t rc
  //           = cudaDeviceCanAccessPeer(&canAccessPeer, cuda_i,cuda_j);
  //         if (rc != cudaSuccess)
  //           throw std::runtime_error("cuda error in cudaDeviceCanAccessPeer: "
  //                                    +std::to_string(rc));
  //         if (!canAccessPeer) {
  //           std::cout << "#dc: warning - could not enable peer access!?"
  //                     << std::endl;
  //           ss << " -";
  //         } else {
  //           cudaSetDevice(cuda_i);
  //           rc = cudaDeviceEnablePeerAccess(cuda_j,0);
  //           ss << " +";
  //         }
  //       }
  //     }
  //     LOG(ss.str()); 
  //   }
  //   cudaSetDevice(restoreActiveDevice);
  //   // reset error
  //   cudaGetLastError();
  // }

    
  /*! pick one of the local nodes' GPUs, base on how many other
    ranks are runnong on that same node. returns the GPU picked
    for this rank. ranks are relative to the 'comm' communicator
    passed to the constructor */
  int Compositor::affinitizeGPU()
  {
    // enablePeerAccess();
    // ------------------------------------------------------------------
    // determine which (world) rank lived on which host, and assign
    // GPUSs
    // ------------------------------------------------------------------
    std::vector<char> sendBuf(MPI_MAX_PROCESSOR_NAME);
    std::vector<char> recvBuf(MPI_MAX_PROCESSOR_NAME*size);
    bzero(sendBuf.data(),sendBuf.size());
    bzero(recvBuf.data(),recvBuf.size());
    int hostNameLen;
    MPI_CALL(Get_processor_name(sendBuf.data(),&hostNameLen));
    std::string hostName = sendBuf.data();
    MPI_CALL(Allgather(sendBuf.data(),sendBuf.size(),MPI_CHAR,
                       recvBuf.data(),/*yes, SENDbuf here*/sendBuf.size(),
                       MPI_CHAR,comm));
    std::vector<std::string> hostNames;
    for (int i=0;i<size;i++) 
      hostNames.push_back(recvBuf.data()+i*MPI_MAX_PROCESSOR_NAME);
    hostName = sendBuf.data();
      
    // ------------------------------------------------------------------
    // count how many other ranks are already on this same node
    // ------------------------------------------------------------------
    MPI_CALL(Barrier(comm));
    int localDeviceID = 0;
    for (int i=0;i<rank;i++) 
      if (hostNames[i] == hostName)
        localDeviceID++;
    MPI_CALL(Barrier(comm));
      
    // ------------------------------------------------------------------
    // assign a GPU to this rank
    // ------------------------------------------------------------------
    int numGPUsOnThisNode;
    CUDA_CALL(GetDeviceCount(&numGPUsOnThisNode));
    if (numGPUsOnThisNode == 0)
      throw std::runtime_error("no GPU on this rank!");
      
    int gpuID = localDeviceID % numGPUsOnThisNode;
    MPI_CALL(Barrier(comm));
      
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpuID);
    std::string gpuName = prop.name;
    this->affinitizedGPU = gpuID;
    printf("#brix.mpi: workers rank #%i on host %s GPU #%i (%s)\n",
           rank,hostName.c_str(),gpuID,gpuName.c_str());
    MPI_CALL(Barrier(comm));
    CUDA_CALL(SetDevice(gpuID));
    // enablePeerAccess();
    return gpuID;
  }
  
  /*! for debugging - save current state to disk */
  void Compositor::save(const std::string &fileName)
  {
    std::ofstream out(fileName,std::ios::binary);
    std::vector<char> tmpMem;

    // uint32_t numPixels = fbSize.x*fbSize.y;
    uint32_t numCompactedFragments
      //      = fullIntCounters[numPixels-1];
      = download(fullIntCounters,numPixelsOrg-1);
    
    out.write((char *)&fbSize,sizeof(fbSize));
    std::vector<uint32_t> h_fullIntCounters(numPixelsOrg);
    download(h_fullIntCounters,fullIntCounters);
                     
    out.write((char *)h_fullIntCounters.data(),numPixelsOrg*sizeof(fullIntCounters[0]));
    out.write((char *)&numCompactedFragments,sizeof(numCompactedFragments));

    std::vector<Fragment> h_compactedFragmentLists(numCompactedFragments);
    download(h_compactedFragmentLists,
             compactedFragmentLists);
    out.write((char *)h_compactedFragmentLists.data(),
              numCompactedFragments*sizeof(compactedFragmentLists[0]));
    std::cout << "SUCCESSFULLY DUMPED FRAME BUFFER, fbsize=" << fbSize.x << "x" << fbSize.y
              << " numFrags=" << numCompactedFragments << std::endl;
  }
  
  /*! for debugging - load output from prev 'save' back in */
  SavedFrags SavedFrags::load(const std::string &fileName)
  {
    SavedFrags result;
    std::ifstream in(fileName,std::ios::binary);
    if (!in.good())
      throw std::runtime_error("cloud not load deep frame buffer from "+fileName);
    in.read((char *)&result.fbSize,sizeof(result.fbSize));
    CUDA_CALL(Malloc(&result.counters,
                     result.fbSize.x*result.fbSize.y*sizeof(*result.counters)));
    std::vector<uint32_t> h_counters(result.fbSize.x*result.fbSize.y);
    in.read((char*)h_counters.data(),//result.counters,
            result.fbSize.x*result.fbSize.y*sizeof(*result.counters));
    CUDA_CALL(Memcpy(result.counters,h_counters.data(),
                     result.fbSize.x*result.fbSize.y*sizeof(*result.counters),
                     cudaMemcpyDefault));
    // std::cout << "first N offsets: ";
    // for (int i=0;i<12;i++)
    //   std::cout << " " << result.counters[i];
    // std::cout << std::endl;
    
    int numFrags;
    in.read((char *)&numFrags,sizeof(numFrags));
    CUDA_CALL(Malloc(&result.fragments,
                     numFrags*sizeof(*result.fragments)));

    std::vector<Fragment> h_frags(numFrags);
    in.read((char*)h_frags.data(),numFrags*sizeof(*result.fragments));
    CUDA_CALL(Memcpy(result.fragments,h_frags.data(),
                     numFrags*sizeof(*result.fragments),
                     cudaMemcpyDefault));
                     
    return result;
  }
    

  // __global__ void compactFragments(Fragment *compactFrags,
  //                                  uint32_t *offsets,
  //                                  uint32_t *counters,
  //                                  int2      fbSize,
  //                                  Fragment *fixedSizeFrags,
  //                                  int       maxFragsPerPixel)
  // {
  //   int jobIdx = threadIdx.x + blockIdx.x*blockDim.x;

  //   int pixelIdx = jobIdx / maxFragsPerPixel;
  //   if (pixelIdx >= fbSize.x*fbSize.y) return;

  //   int fragIdx = jobIdx % maxFragsPerPixel;
  //   if (fragIdx >= counters[pixelIdx]) return;

  //   compactFrags[offsets[pixelIdx]+fragIdx]
  //     = fixedSizeFrags[pixelIdx*maxFragsPerPixel+fragIdx];
  // }
  


  __global__
  void compactFragmentsKernel(Fragment *compacted,
                              uint32_t *offsets,
                              const Fragment *fragments,
                              int numFragsPerPixel,
                              int numPixelsPadded)
  {
    int pixelID = threadIdx.x + blockIdx.x*blockDim.x;
    if (pixelID >= numPixelsPadded) return;

    // int fbSizeX = 320;
    // int fbSizeY = 200;
    bool dbg = 0;//pixelID == 0;//0; //pixelID == (fbSizeX*fbSizeY/2+fbSizeX/2);
    
    uint32_t end = offsets[pixelID];
    uint32_t begin
      = (pixelID == 0)
      ? 0
      : offsets[pixelID-1];
    int count = end-begin;
    const Fragment *in = fragments+pixelID*numFragsPerPixel;

    if (dbg) printf("copying frags: begin/end = %i..%i, #=%i\n",begin,end,count);
    
    for (int i=0;i<count;i++) {
      compacted[begin+i] = in[i];

      float4 _in = in[i].getRGBA();
      if (dbg) printf("frag %i: rgb= %f %f %f a=%f z=%f\n",i,
                      _in.x,
                      _in.y,
                      _in.z,
                      _in.w,
                      float(in[i].z));
    }
  }

  // __global__
  // void pageInKernel(uint32_t *fullIntCounters,
  //                   int numPixels)
  // {
  //   int pixelID = threadIdx.x + blockIdx.x*blockDim.x;
  //   if (pixelID >= numPixels) return;

  //   if (fullIntCounters[pixelID] == uint32_t(-1))
  //     printf("bla\n");
  // }

  __global__
  void compositeKernel(uint32_t *compositedColor,
                       const uint32_t *compOffsets,
                       Fragment *incomingFragments,
                       uint32_t numPixelsOnThisRank,
                       int /*mpi group size*/size)
  {
    int pixelIdx = threadIdx.x + blockIdx.x*blockDim.x;
    if (pixelIdx >= numPixelsOnThisRank) return;

    bool dbg = 0;//pixelIdx == 0;
    
    float alpha = 0.f;
    float3 color = make_float3(0.f,0.f,0.f);
    while (1) {
      Fragment *nextClosestFragment = nullptr;
      // find next closest fragment:
      for (int fromRank=0;fromRank<size;fromRank++) {
        int idx = fromRank*numPixelsOnThisRank+pixelIdx;
        int begin = idx?compOffsets[idx-1]:0;
        int end = compOffsets[idx];
        for (int i=begin;i<end;i++) {
          Fragment *fragment = &incomingFragments[i];
          if (nextClosestFragment == nullptr ||
              float(fragment->z) < float(nextClosestFragment->z))
            nextClosestFragment = fragment;
        }
      }
      if (nextClosestFragment == nullptr || float(nextClosestFragment->z) >= 1e20f)
        break;

      float4 fragColor = nextClosestFragment->getRGBA();
      if (dbg) printf("fragColor %f %f %f %f\n",
                      fragColor.x,
                      fragColor.y,
                      fragColor.z,
                      fragColor.w);
      color = color
        +  (1.f-alpha)
        // *  fragColor.w
        *  (const float3&)fragColor;
      // float3((float)nextClosestFragment->r,
      //          (float)nextClosestFragment->g,
      //          (float)nextClosestFragment->b);
      alpha += (1.f-alpha)*fragColor.w;

      nextClosestFragment->z = 1e20f;
    }
    compositedColor[pixelIdx] = make_rgba(color);
  }
    
  inline std::ostream &operator<<(std::ostream &o,
                                  const Fragment &frag)
  {
    o << "{z=" << float(frag.z)
      << ",rgb=("<<(float)frag.r << "," << (float)frag.g << "," << (float)frag.b << ")"
      <<",a="<<frag.getRGBA().w<<"}";
    return o;
  }

  inline int computeCountersPerByte(uint32_t maxFragsPerPixel)
  {
    assert(maxFragsPerPixel <= 255);
    if (maxFragsPerPixel < 4) return 4;
    else if (maxFragsPerPixel < 16) return 2;
    else return 1;
  }
  
  
  Compositor::Compositor(int maxFragsPerPixel,
                         MPI_Comm comm)
    : comm(comm),
      maxFragsPerPixel(maxFragsPerPixel),
      numCountersPerByte(computeCountersPerByte(maxFragsPerPixel))
  {
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
    ProfPrinter::rank = rank;
  }
    

  DeviceInterface Compositor::prepare()
  {
    if (fbSize.x <= 0 || fbSize.y <= 0)
      throw std::runtime_error("invalid or un-specified frame buffer size in dc library..."
                               "did you forget a 'DeepCompositor::resize()'?");
    int oldDevice = -1;
    if (affinitizedGPU >= 0) {
      CUDA_CALL(GetDevice(&oldDevice));
      CUDA_CALL(SetDevice(affinitizedGPU));
    }

    CUDA_CALL(Memset(fullIntCounters,0,
                     numPixelsPadded//fbSize.x*fbSize.y
                     *sizeof(int)));

    deviceData.counters = fullIntCounters;
    deviceData.fbSize = fbSize;
    deviceData.maxFragsPerPixel = maxFragsPerPixel;
    deviceData.fragments = fixedSizeFragmentLists;
    
    if (oldDevice >= 0) {
      CUDA_CALL(SetDevice(oldDevice));
    }

    return deviceData;
  }


  // ==================================================================
  // the actual compositing code, independent of interface
  // ==================================================================
  __global__ void compactCountersKernel(uint8_t *loBitCounters,
                                        uint32_t *fullIntCounters,
                                        int numLoBitCounters,
                                        int numCountersPerByte)
  {
    int threadID = threadIdx.x+blockIdx.x*blockDim.x;
    if (threadID >= numLoBitCounters) return;

    // bool dbg = threadID == 13;
    
    uint8_t loBitCounter = 0;
    int bitsPerCounter = 8 / numCountersPerByte;
    // if (dbg) printf("bits %i\n",bitsPerCounter);
    
    for (int i=0;i<numCountersPerByte;i++) {
      int pixelID = numCountersPerByte*threadID+i;
      uint32_t counter = fullIntCounters[pixelID];
      loBitCounter += (counter << (i*bitsPerCounter));
      // if (dbg) printf("ctr %i shift %i res %i\n",
      //                 counter,i*bitsPerCounter,loBitCounter);
    }
    loBitCounters[threadID] = loBitCounter;
  }

  __global__ void uncompressCountersKernel(uint32_t *fullIntCounters,
                                           uint8_t *loBitCounters,
                                           uint32_t numLoBitCounters,
                                           uint32_t numCountersPerByte)
  {
    int threadID = threadIdx.x+blockIdx.x*blockDim.x;
    if (threadID >= numLoBitCounters) return;

    uint8_t bits = loBitCounters[threadID];
    int bitsPerCounter = 8 / numCountersPerByte;
    uint32_t mask = (1<<bitsPerCounter)-1;
    for (int i=0;i<numCountersPerByte;i++) {
      fullIntCounters[numCountersPerByte*threadID+i]
        = (bits >> (i*bitsPerCounter)) & mask;
    }
  }

  void Compositor::finish(uint32_t *whereToWriteFinalPixels)
  {
    CUDA_SYNC_CHECK();
    if (affinitizedGPU < 0)
      std::cout << "WARNING: NOT AFFINITIZED!" << std::endl;
      
    // double t0 = getCurrentTime();
    prof_all.enter();
    
    size_t stat_numBytesCountersOut = 0;
    size_t stat_numBytesCountersIn = 0;
    size_t stat_numBytesFragsOut = 0;
    size_t stat_numBytesFragsIn = 0;
    
    int oldDevice = -1;
    if (affinitizedGPU >= 0) {
      CUDA_CALL(GetDevice(&oldDevice));
      CUDA_CALL(SetDevice(affinitizedGPU));
    }

    CUDA_SYNC_CHECK();
    const uint32_t numPixelsOnThisRank
      = pixelEnd(rank) - pixelBegin(rank);
    // actual, non padded pixels on this rank:
    // const uint32_tint numPixelsOnThisRankOrg
    //   = std::min(pixelEnd(rank),numPixelsOrg) - std::min(pixelBegin(rank),numPixelsOrg);
    //      (rank+1)*numPixels/size
    //      - (rank+0)*numPixels/size;
    
    // ------------------------------------------------------------------
    // compact counters: this turns the 32-bit uint counters to 2 or
    // 4-bit values within the compactCounters[] array
    // ------------------------------------------------------------------
    prof_cudaCompactCounters.enter();
    {
      // for (int i=0;i<30;i++)
      //   std::cout << fullIntCounters[i] << " ";
      // std::cout << std::endl;
      uint32_t numCCBytes = numPixelsPadded / numCountersPerByte;
        // = divRoundUp(numPixels(),numCountersPerByte);
      const uint32_t blockSize = 128;
      const uint32_t numBlocks = divRoundUp(numCCBytes,blockSize);
      compactCountersKernel<<<numBlocks,blockSize>>>
        (lowBitCounters,fullIntCounters,numCCBytes,numCountersPerByte);
      
      CUDA_SYNC_CHECK();
      // for (int i=0;i<30;i++)
      //   std::cout << (int*)(long)lowBitCounters[i] << " ";
      // std::cout << std::endl;
    }
    prof_cudaCompactCounters.leave();
    
    // ------------------------------------------------------------------
    // compact fragments - eliminate unused fragements by compacting
    // fixed-list length into compact sequential lists; this also
    // turns the counters[] array into a offsets[] array
    // ------------------------------------------------------------------
    {
      // for (int i=0;i<30;i++)
      //   std::cout << fullIntCounters[i] << " ";
      // std::cout << std::endl;


      // {
      //   const uint32_t blockSize = 128;
      //   const uint32_t numBlocks = divRoundUp(numPixels(),blockSize);
      //   pageInKernel<<<numBlocks,blockSize>>>(fullIntCounters,numPixels());
      //   CUDA_SYNC_CHECK();
      // }        
      prof_cudaCompactFrags_scan.enter();
      // force padded counters to 0.
      CUDA_CALL(Memset(fullIntCounters+numPixelsOrg,0,
                       (numPixelsPadded-numPixelsOrg)*sizeof(int)));
#if 1
      
      // Declare, allocate, and initialize device-accessible pointers for input and output
      int  num_items = numPixelsPadded;      // e.g., 7
      int  *d_in = (int *)fullIntCounters;          // e.g., [8, 6, 7, 5, 3, 0, 9]
      int  *d_out  = 0;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
      CUDA_CALL(Malloc((void **)&d_out,num_items*sizeof(int)));
      //...
      // Determine temporary device storage requirements for inclusive prefix sum
      void     *d_temp_storage = NULL;
      size_t   temp_storage_bytes = 0;
      cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
      // Allocate temporary storage for inclusive prefix sum
      cudaMalloc(&d_temp_storage, temp_storage_bytes);
      // Run inclusive prefix sum
      cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
      cudaMemcpy(d_in,d_out,num_items*sizeof(int),cudaMemcpyDefault);
      cudaFree(d_out);
      cudaFree(d_temp_storage);
#else
      thrust::inclusive_scan
        (thrust::device,
         thrust::device_pointer_cast<uint32_t>(fullIntCounters),
         thrust::device_pointer_cast<uint32_t>(fullIntCounters+numPixelsPadded),
         thrust::device_pointer_cast<uint32_t>(fullIntCounters));
#endif
      CUDA_SYNC_CHECK();
      prof_cudaCompactFrags_scan.leave();

      prof_cudaCompactFrags_copy.enter();
      const uint32_t blockSize = 128;
      const uint32_t numBlocks = divRoundUp(numPixelsPadded,blockSize);
      compactFragmentsKernel<<<numBlocks,blockSize>>>
        (compactedFragmentLists, fullIntCounters,
         fixedSizeFragmentLists, maxFragsPerPixel,
         numPixelsPadded);
      
      CUDA_SYNC_CHECK();
      prof_cudaCompactFrags_copy.leave();
      // PING;
      // for (int i=0;i<30;i++)
      //   std::cout << fullIntCounters[i] << " ";
      // std::cout << std::endl;
    }

    if (dumpFileName != "")
      save(dumpFileName);


    // ------------------------------------------------------------------
    // use OUR offsets - before we re-use the fullint array as
    // receiving buffer - to compute outgoing offsets/countes
    // ------------------------------------------------------------------

    std::vector<int> numFragmentsTo(size);
    std::vector<int> numFragmentsFrom(size);
    std::vector<int> numBytesTo(size);
    std::vector<int> numBytesFrom(size);
    std::vector<int> ofsBytesTo(size);
    std::vector<int> ofsBytesFrom(size);
    int              sumBytesReceiving = 0;
    int sumBytesTo = 0;
    int sumBytesFrom = 0;
    int sumFragsTo = 0;
    int sumFragsFrom = 0;
    // -------------------------------------------------------
    // compute the *number* of fragments sent to / received from any
    // other node, so we can set up the MPI_Alltoall
    // -------------------------------------------------------
    for (int node=0;node<size;node++) {
      uint32_t node_pixel_begin = pixelBegin(node); //(node+0)*numPixels / size;
      uint32_t node_pixel_end   = pixelEnd(node); //(node+1)*numPixels / size;
      numFragmentsTo[node]
        = (node_pixel_end  ?download(fullIntCounters,node_pixel_end  -1):0)
        - (node_pixel_begin?download(fullIntCounters,node_pixel_begin-1):0);
      // numFragmentsTo[node]
      //   = (node_pixel_end  ?fullIntCounters[node_pixel_end  -1]:0)
      //   - (node_pixel_begin?fullIntCounters[node_pixel_begin-1]:0);
      numBytesTo[node]
        = numFragmentsTo[node] * sizeof(Fragment);

      ofsBytesTo[node] = sumBytesTo;
      sumFragsTo += numFragmentsTo[node];
      sumBytesTo += numBytesTo[node];

      if (node != rank)
        stat_numBytesFragsOut += numBytesTo[node];
    }
    // stat_numBytesFragsOut += sumBytesTo;


    


    
    
    // ------------------------------------------------------------------
    // compute offsets and counts (relative to compressed counters
    // array) of what we want to send _out_
    // ------------------------------------------------------------------
    std::vector<int> numCounterBytesTo(size);
    std::vector<int> ofsCounterBytesTo(size);
    uint32_t ofsTo = 0;
    for (int node=0;node<size;node++) {
      // range of pixels (from us) that _this_ node is responsible for
      numCounterBytesTo[node] = (pixelEnd(node) - pixelBegin(node))/numCountersPerByte;
      ofsCounterBytesTo[node] = ofsTo;
      ofsTo   += numCounterBytesTo[node];
      if (node != rank)
        stat_numBytesCountersOut += numCounterBytesTo[node];
    }
    // stat_numBytesCountersOut += ofsTo;
    
    // ------------------------------------------------------------------
    // compute offsets and counts (relative to compressed counters
    // array) of what we want to _receive_ from others
    // ------------------------------------------------------------------
    std::vector<int> numCounterBytesFrom(size);
    std::vector<int> ofsCounterBytesFrom(size);
    uint32_t sumCounterBytesReceiving = 0;
    uint32_t ofsFrom = 0;
    for (int node=0;node<size;node++) {
      // range of pixels (from this node) that _we_ are responsible for
      numCounterBytesFrom[node] = numPixelsOnThisRank / numCountersPerByte;
      sumCounterBytesReceiving += numCounterBytesFrom[node];

      ofsCounterBytesFrom[node] = ofsFrom;
      ofsFrom += numCounterBytesFrom[node];
      // if (rank == 0) { PRINT(node); PRINT(ofsFrom); PRINT(ofsCounterBytesFrom[node]); }
      if (node != rank)
        stat_numBytesCountersIn += numCounterBytesFrom[node];
    }
    // stat_numBytesCountersIn += ofsFrom;
    // now alloc and exchange
    uint8_t *recvCompactCounters = nullptr;
    // printf("sumCounterBytesReceiving(%i) = %i\n",
    //        rank,sumCounterBytesReceiving);
    prof_exchangeCounters.enter();
    CUDA_CALL(Malloc(&recvCompactCounters,
                     sumCounterBytesReceiving));

#if 0
    for (int i=0;i<size;i++) {
      if (rank == i)
        printf("ctrs(%i) 0:IN #%i @%i OUT #%i @%i  1:IN #%i @%i OUT #%i @%i  2:IN #%i @%i OUT #%i @%i\n",
               rank,
               numCounterBytesFrom[0],
               ofsCounterBytesFrom[0],
               numCounterBytesTo[0],
               ofsCounterBytesTo[0],
               numCounterBytesFrom[1],
               ofsCounterBytesFrom[1],
               numCounterBytesTo[1],
               ofsCounterBytesTo[1],
               numCounterBytesFrom[2],
               ofsCounterBytesFrom[2],
               numCounterBytesTo[2],
               ofsCounterBytesTo[2]);
      MPI_CALL(Barrier(comm));
    }
#endif
    MPI_CALL(Alltoallv(lowBitCounters,numCounterBytesTo.data(),
                       ofsCounterBytesTo.data(),MPI_BYTE,
                       recvCompactCounters,numCounterBytesFrom.data(),
                       ofsCounterBytesFrom.data(),MPI_BYTE,
                       comm));
    CUDA_SYNC_CHECK();
    prof_exchangeCounters.leave();


    prof_decodeCounters.enter();
    // ------------------------------------------------------------------
    // uncompress the received compressed counters to the 'real'
    // counters ... careful, this overwrites the fullint array that we
    // so far used for as offset array
    // ------------------------------------------------------------------
    {
      uint32_t blockSize = 128;
      uint32_t numBlocks = divRoundUp(sumCounterBytesReceiving,blockSize);
      uncompressCountersKernel<<<numBlocks,blockSize>>>
        (fullIntCounters,recvCompactCounters,
         sumCounterBytesReceiving,numCountersPerByte);
    }
    CUDA_CALL(Free(recvCompactCounters));
    prof_decodeCounters.leave();

    // ------------------------------------------------------------------
    // prefix sum so we can look up each pixels' fragment begin in a
    // single array
    // ------------------------------------------------------------------

    prof_computeOffsets.enter();
    const int numCountersReceived = numRanks()*computeNumPixelsOnThisRank();
#if 1
      
      // Declare, allocate, and initialize device-accessible pointers for input and output
      int  num_items = numCountersReceived;      // e.g., 7
      int  *d_in = (int *)fullIntCounters;          // e.g., [8, 6, 7, 5, 3, 0, 9]
      int  *d_out  = 0;         // e.g., [ ,  ,  ,  ,  ,  ,  ]
      CUDA_CALL(Malloc((void **)&d_out,num_items*sizeof(int)));
      //...
      // Determine temporary device storage requirements for inclusive prefix sum
      void     *d_temp_storage = NULL;
      size_t   temp_storage_bytes = 0;
      cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
      // Allocate temporary storage for inclusive prefix sum
      cudaMalloc(&d_temp_storage, temp_storage_bytes);
      // Run inclusive prefix sum
      cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
      cudaMemcpy(d_in,d_out,num_items*sizeof(int),cudaMemcpyDefault);
      cudaFree(d_out);
      cudaFree(d_temp_storage);
#else
    thrust::inclusive_scan
      (thrust::device,
       thrust::device_pointer_cast<uint32_t>(fullIntCounters),
       thrust::device_pointer_cast<uint32_t>(fullIntCounters+numCountersReceived),
       thrust::device_pointer_cast<uint32_t>(fullIntCounters));
#endif
    CUDA_SYNC_CHECK();
    prof_computeOffsets.leave();

    prof_exchangeFrags.enter();
    for (int node=0;node<size;node++) {
      int node_comp_begin = (node+0)*numPixelsOnThisRank;
      int node_comp_end   = (node+1)*numPixelsOnThisRank;
      // numFragmentsFrom[node]
      //   = (node_comp_end  ?fullIntCounters[node_comp_end  -1]:0)
      //   - (node_comp_begin?fullIntCounters[node_comp_begin-1]:0);
      numFragmentsFrom[node]
        = (node_comp_end  ?download(fullIntCounters,node_comp_end  -1):0)
        - (node_comp_begin?download(fullIntCounters,node_comp_begin-1):0);
      numBytesFrom[node]
        = numFragmentsFrom[node] * sizeof(Fragment);
      ofsBytesFrom[node] = sumBytesFrom;
      sumFragsFrom += numFragmentsFrom[node];
      sumBytesFrom += numBytesFrom[node];
      sumBytesReceiving += numBytesFrom[node];
      if (node != rank)
        stat_numBytesFragsIn += numBytesFrom[node];
    }
    // stat_numBytesFragsIn += sumBytesReceiving;
    
    // for (int node=0;node<size;node++) {
    //   uint32_t node_pixel_begin = pixelBegin(node); //(node+0)*numPixels / size;
    //   uint32_t node_pixel_end   = pixelEnd(node); //(node+1)*numPixels / size;
    //   numFragmentsTo[node]
    //     = (node_pixel_end  ?fullIntCounters[node_pixel_end  -1]:0)
    //     - (node_pixel_begin?fullIntCounters[node_pixel_begin-1]:0);
    //   numBytesTo[node]
    //     = numFragmentsTo[node] * sizeof(Fragment);

    //   int node_comp_begin = (node+0)*numPixelsOnThisRank;
    //   int node_comp_end   = (node+1)*numPixelsOnThisRank;
    //   numFragmentsFrom[node]
    //     = (node_comp_end  ?fullIntCounters[node_comp_end  -1]:0)
    //     - (node_comp_begin?fullIntCounters[node_comp_begin-1]:0);
    //   numBytesFrom[node]
    //     = numFragmentsFrom[node] * sizeof(Fragment);
        
    //   ofsBytesTo[node] = sumBytesTo;
    //   ofsBytesFrom[node] = sumBytesFrom;
    //   sumFragsFrom += numFragmentsFrom[node];
    //   sumFragsTo += numFragmentsTo[node];
    //   sumBytesTo += numBytesTo[node];
    //   sumBytesFrom += numBytesFrom[node];
    //   sumBytesReceiving += numBytesFrom[node];
    // }
   // MPI_CALL(Barrier(comm));

    // #if PRINT_STATS
    //     {
    //       printf("(%i) num pixels %i, frags in/out %s/%s bytes %s/%s, sumfrags/pix %f, sumbytes/pix %f\n",
    //              rank,numPixels,
    //              prettyNumber(sumFragsFrom).c_str(),
    //              prettyNumber(sumFragsTo).c_str(),
    //              prettyNumber(sumBytesFrom).c_str(),
    //              prettyNumber(sumBytesTo).c_str(),
    //              (sumFragsFrom+sumFragsTo)/float(numPixels),
    //              (sumFragsFrom+sumFragsTo+numPixels*sizeof(*deviceData.counters))/float(numPixels));
    //     }
    // #endif
    // ------------------------------------------------------------------
    // now, allocate the memory for all fragments we'll receive, and
    // mpi-exchange with other nodes
    // ------------------------------------------------------------------
    Fragment *incomingFragments = fixedSizeFragmentLists;

#if 0
    for (int i=0;i<size;i++) {
      if (rank == i)
        printf("ctrs(%i) to %i/%i %i/%i %i/%i from %i/%i %i/%i %i/%i\n",
               rank,
               ofsBytesTo[0],
               numBytesTo[0],
               ofsBytesTo[1],
               numBytesTo[1],
               ofsBytesTo[2],
               numBytesTo[2],
               ofsBytesFrom[0],
               numBytesFrom[0],
               ofsBytesFrom[1],
               numBytesFrom[1],
               ofsBytesFrom[2],
               numBytesFrom[2]);
      MPI_CALL(Barrier(comm));
    }
#endif

    
    MPI_CALL(Alltoallv(compactedFragmentLists,numBytesTo.data(),
                       ofsBytesTo.data(),MPI_BYTE,
                       incomingFragments,numBytesFrom.data(),
                       ofsBytesFrom.data(),MPI_BYTE,
                       comm));
    prof_exchangeFrags.leave();
    CUDA_SYNC_CHECK();
    // MPI_CALL(Barrier(comm));
    // ==================================================================
    // finally, have all fragments ... composite
    // ==================================================================
    prof_cudaCompositing.enter();
    uint32_t *compositedColor = nullptr;
    CUDA_CALL(Malloc(&compositedColor,
                     numPixelsOnThisRank*sizeof(*compositedColor)));
    compositeKernel<<<divRoundUp((int)numPixelsOnThisRank,128),128>>>
      (compositedColor,fullIntCounters,
       incomingFragments,numPixelsOnThisRank,size);
    CUDA_SYNC_CHECK();
    prof_cudaCompositing.leave();

    
    // ==================================================================
    // finally, have composited colors on this rank - gather at master
    // ==================================================================

    if (rank == 0) {
      prof_finalAssemble.enter();
      std::vector<MPI_Request> requests(size);
      CUDA_CALL(Memcpy(whereToWriteFinalPixels,
                       compositedColor,
                       std::min(numPixelsOnThisRank,numPixelsOrg)*sizeof(*compositedColor),
                       cudaMemcpyDefault));
      // std::copy(compositedColor,
      //           compositedColor+numPixelsOnThisRank,
      //           whereToWriteFinalPixels);
      for (int node=1;node<size;node++) {
        int begin = pixelBegin(node);//(node+0)*fbSize.x*fbSize.y / size;
        int end   = std::min(numPixelsOrg,pixelEnd(node));//(node+1)*fbSize.x*fbSize.y / size;
        if (begin < end)
          MPI_CALL(Irecv(whereToWriteFinalPixels+begin,
                         (end-begin)*sizeof(*whereToWriteFinalPixels),
                         MPI_BYTE,node,0,comm,&requests[node]));
      }
      MPI_CALL(Waitall(size-1,requests.data()+1,MPI_STATUS_IGNORE));
      prof_finalAssemble.leave();
    } else {
      prof_finalAssemble.enter();
      int begin = pixelBegin(rank);//(node+0)*fbSize.x*fbSize.y / size;
      int end   = std::min(numPixelsOrg,pixelEnd(rank));//(node+1)*fbSize.x*fbSize.y / size;
      MPI_CALL(Send(compositedColor,
                    (end-begin)/*numPixelsOnThisRankOrg*/*sizeof(*whereToWriteFinalPixels),
                    MPI_BYTE,0,0,comm));
      prof_finalAssemble.leave();
    }
    prof_all.leave();
    // CUDA_SYNC_CHECK();

    //==================================================================
    // aaaand..... done. free all local mem
    //==================================================================
    // CUDA_CALL(Free(compCounters));
    // CUDA_CALL(Free(compOffsets));
    // CUDA_CALL(Free(incomingFragments));
    CUDA_CALL(Free(compositedColor));
    if (oldDevice >= 0) {
      CUDA_CALL(SetDevice(oldDevice));
    }
    CUDA_SYNC_CHECK();

    static int g_pass = 0;
    if (g_pass++ == 0)
      printf("rank %i: bytes in/out for counters: %.2fMB/out %.2fMB, frags: %.2fMB/%.2fMB, total: %.2fMB/%.2fMB \n",
             rank,
             float(stat_numBytesCountersIn)/(1024*1024),
             float(stat_numBytesCountersOut)/(1024*1024),
             float(stat_numBytesFragsIn)/(1024*1024),
             float(stat_numBytesFragsOut)/(1024*1024),
             float(stat_numBytesCountersIn+stat_numBytesFragsIn)/(1024*1024),
             float(stat_numBytesCountersOut+stat_numBytesFragsOut)/(1024*1024)
             );
    // double t1 = getCurrentTime();
    // if (rank == 0)
    //   std::cout << "time in finish " << 1000*(t1-t0) << " ms" << std::endl;
  }
      
}
