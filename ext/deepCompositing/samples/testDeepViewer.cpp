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

#include <cuda_runtime.h>
#include <mpi.h>
#include "deepImageRenderer.h"
#include "testRenderer.h"
#include <vector>
#include "samples/common/owlViewer/OWLViewer.h"
#include <chrono>

// #define STB_IMAGE_WRITE_IMPLEMENTATION 1
# include "stb/stb_image.h"
# include "stb/stb_image_write.h"

namespace dctest {

  Camera camera;
  int rank, size;
  
  int testCubeRes = 8;
  
  
  typedef enum { CMD_TERMINATE, CMD_CAMERA, CMD_RENDER, CMD_RESIZE } Command;
  
  struct Viewer : public owl::viewer::OWLViewer
  {
    Viewer(dc::Compositor &compositor)
      : compositor(compositor)
    {
      box3f bounds(vec3f(0.f),vec3f(1.f));
      const float worldScale = length(bounds.span());
      
      this->enableFlyMode();
      this->enableInspectMode(bounds);
      this->setWorldScale(owl::length(bounds.span()));

      this->setCameraOrientation(/*origin   */
                                 bounds.center()
                                 + vec3f(-.4f, .7f, +1.5f) * bounds.span(),
                                 /*lookat   */bounds.center(),
                                 /*up-vector*/vec3f(0.f, 1.f, 0.f),
                                 /*fovy(deg)*/40.f);
       // --camera -0.2173137367 1.396072865 2.363779068 0.5380306244 0.4386408925 0.5505136251 0 1 0 
      this->setCameraOrientation(vec3f(-.217,1.4,2.36),vec3f(0.54,0.44,0.55),
                                 /*up-vector*/vec3f(0.f, 1.f, 0.f),
                                 /*fovy(deg)*/40.f);
      
      this->setWorldScale(worldScale);

      glfwSetWindowSize(handle,1024,1024);
      // glfwSetWindowSize(handle,2488,1016);
    }
    
    virtual void render() override
    {
      if (fbSize.x < 0) return;

      std::chrono::steady_clock::time_point
        begin_render = std::chrono::steady_clock::now();

      double t0 = getCurrentTime();
      
      int cmd = CMD_RENDER;
      MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
      

      // ------------------------------------------------------------------
      // render one frame
      // ------------------------------------------------------------------
      compositor.resize(fbSize.x,fbSize.y);
      dc::DeviceInterface interface = compositor.prepare();
      // dc::TwoStageInterface interface(compositor);
      // dctest::countFragments(interface.firstStage({fbSize.x,fbSize.y}),
      //                        dctest::camera, rank, size);
      // dctest::renderFragments(interface.secondStage(),
      //                         dctest::camera, rank,size, fakeDepth);
      // dctestImage::countFragments(interface.firstStage({fbSize.x,fbSize.y}),
      //                             (dctestImage::Camera &)camera, rank,  size,
      //                             "/local/sub-yetanother");
      // dctestImage::renderFragments(interface.secondStage(),
      //                             (dctestImage::Camera &)camera, rank, size,
      //                             "/local/sub-yetanother");
      dctest::renderFragments(testCubeRes,interface,
                              dctest::camera, rank,size);
      cudaDeviceSynchronize();

      // ------------------------------------------------------------------
      // composite frame
      // ------------------------------------------------------------------
      compositor.finish(fbPointer);

      std::chrono::steady_clock::time_point
        end_render = std::chrono::steady_clock::now();

      double t1 = getCurrentTime();
      std::cout << "ms per frame " << (t1-t0)*1000 << std::endl;

      double fps = 1000.f/std::chrono::duration_cast<std::chrono::milliseconds>(end_render-begin_render).count();
      setTitle("Deep Compositing Test Viewer ("+prettyDouble(fps)+" FPS)");
    }
    
    void cameraChanged() override
    {
      const auto cam = getSimplifiedCamera();

      dctest::camera.org = cam.lens.center;
      dctest::camera.dir_00 = cam.screen.lower_left;
      dctest::camera.dir_du = cam.screen.horizontal / float(fbSize.x);
      dctest::camera.dir_dv = cam.screen.vertical / float(fbSize.y);
      
      int cmd = CMD_CAMERA;
      MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast(&dctest::camera,sizeof(dctest::camera),MPI_BYTE,0,MPI_COMM_WORLD);
    }
    
    void resize(const vec2i &newSize) override
    {
      OWLViewer::resize(newSize);
      
      int cmd = CMD_RESIZE;
      MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
      MPI_Bcast((void *)&newSize,sizeof(newSize),MPI_BYTE,0,MPI_COMM_WORLD);

      // camera pixel vectors depend on screen res, so update camera,
      // too:
      cameraChanged();
    }

    void key(char key, const vec2i &where) override
    {
      switch(key) {
      case 'C': {
        auto &fc = getCamera();
        std::cout << "(C)urrent camera:" << std::endl;
        std::cout << "- from :" << fc.position << std::endl;
        std::cout << "- poi  :" << fc.getPOI() << std::endl;
        std::cout << "- upVec:" << fc.upVector << std::endl; 
        std::cout << "- frame:" << fc.frame << std::endl;
        std::cout.precision(10);
        std::cout << "cmdline: --camera "
                  << fc.position.x << " "
                  << fc.position.y << " "
                  << fc.position.z << " "
                  << fc.getPOI().x << " "
                  << fc.getPOI().y << " "
                  << fc.getPOI().z << " "
                  << fc.upVector.x << " "
                  << fc.upVector.y << " "
                  << fc.upVector.z << " "
          // << "--fov " << cmdline.camera.fov
                  << std::endl;
      } break;
      case '!': {
        screenShot("fun3d-viewer.png");
      } break;
      default:
        OWLViewer::key(key,where);
        break;
      };
    }
    dc::Compositor &compositor;
  };
  
  extern "C" int main(int ac, char **av)
  {
    MPI_Init(&ac,&av);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    dc::Compositor compositor;
    compositor.affinitizeGPU();
    
    if (rank == 0) {
      
      // vec3f lookAt = { .5,.5,.5 };
      // vec3f lookFrom = { -1,-2,-3 };
      // vec3f lookUp   = { 0,1,0 };
      // float zoom = 2.f;

      vec3f from { 0.05217219144, 0.191569522, 0.5715021491 };
      vec3f at   {  0.5941146612, 1.011739969, 0.5692445636 };
      vec3f up   { 0, 1, 0 };

      Viewer viewer(compositor);

      viewer.setCameraOrientation(/*origin   */from,
                                  /*lookat   */at,
                                  /*up-vector*/up,
                                  /*fovy(deg)*/60.f);
      
      viewer.showAndRun();
    }
    else {
      int cmd;
      vec2i  fbSize = { -1,-1 };
      do {
        MPI_Bcast(&cmd,1,MPI_INT,0,MPI_COMM_WORLD);
        switch (cmd) {
        case CMD_TERMINATE: {
        } break;
        case CMD_RESIZE: {
          MPI_Bcast(&fbSize,sizeof(fbSize),MPI_BYTE,0,MPI_COMM_WORLD);
        } break;
        case CMD_CAMERA: {
          MPI_Bcast(&camera,sizeof(camera),MPI_BYTE,0,MPI_COMM_WORLD);
        } break;
        case CMD_RENDER: {
          // ------------------------------------------------------------------
          compositor.resize(fbSize.x,fbSize.y);
          dc::DeviceInterface interface = compositor.prepare();
          dctest::renderFragments(testCubeRes, interface, camera, rank,size);

          cudaDeviceSynchronize();
          uint32_t *fbPointer = 0;
          compositor.finish(fbPointer);
        } break;
        default:
          std::cout << "UNKNOWN CMD " << cmd << std::endl;
        };
      } while (cmd != CMD_TERMINATE);
    }
    // ------------------------------------------------------------------
    // done rendering
    // ------------------------------------------------------------------
    MPI_Finalize();
    return 0;
  }
}
