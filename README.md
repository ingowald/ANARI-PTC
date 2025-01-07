# ANARI_PTC (Pass-Through Compositing) Device

ANARI (https://registry.khronos.org/ANARI/specs/1.0/ANARI-1.0.html) is
a Khonos Group-maintained API for cross-platform rendering.  The repo
provides an implementation that allows ANARI to be used for
compositing-based data-parallel rendering, where multiple different
MPI ranks each render their local data, in a ANARI-constent way. 

For more details, plealse see

```
Standardized Data-Parallel Rendering Using ANARI
Ingo Wald, Stefan Zellmann, Jefferson Amstutz, Qi Wu, Kevin Griffin, Milan Jaros, Stefan Wesner
IEEE Symposium on Large Data Analysis and Visualization (LDAV), 2024
```
(free PDF version available here https://arxiv.org/abs/2407.00179)

# Dependencies

This project requires
- a built and `make install`ed `ANARI_SDK`
  (https://github.com/KhronosGroup/ANARI-SDK)
- cmake, c compiler, etc (which you already need for the ANARI SDK,
  anyway)
- when buiding the *optional* (GPU-compositing version): `CUDA` (if CUDA
  is disabled or cannot be found the project will automatically fall
  back to CPU compositing)
- when building the CPU version: `tbb`
- MPI development packages. When built with CUDA support you'll need a
  CUDA-aware MPI build; for the cpu-only version any MPI build should
  do.

# Building 

- Make sure to build and install the ANARI_SDK
- `mkdir buildDir`
- `cd buildDir`
- `cmake ..` (for default build)
or
- `cmake -DPTC_DISABLE_CUDA .. (for explicitly cpu-only build

# Installing

- just regular cmake `make install` ...

# Using ANARI_PTC

Assuming you have a data-parallel application that already has a ANARI
rendering path, just use this with `ANARI_LIBRARY` set to `ptc`

# Contributors

Though the project lives under my github namespace (and all files bear
my copyright headers), much of the latest version of the code is
actually entirely credit-due to Jeff Amstutz - who always takes
whatever duct-tape mess i've once written, burns it in a bonfire, and
rewrites it in a much cleaner and more reliable way.

# Related Projects

- ANARI-SDK (https://github.com/KhronosGroup/ANARI-SDK)
- barney (https://github.com/ingowald/barney)
- pynari (https://github.com/ingowald/pynari)
- VisRTX (https://github.com/NVIDIA/VisRTX)

# Version History

## v1.0.0

- first working version

## v1.1.0

- added CI build
- fixed various config issues for different cmake/linux/gcc versions
- now properly builds even on systems that do not have cuda installed
