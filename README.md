## GPUFR

GPUFR (name tbd) is a project created to investigate the reconstruction of polynomials using GPGPU devices. It is currently implemented in CUDA and C/C++.

### Dependencies

The core depencies of the project are:

- CMake version 3.20+
- nvcc version 12.5+

However, to run the testing suite you will also need the FLINT library v3.1.2+, and all of it's dependencies. FLINT can be found [here](https://github.com/flintlib/flint)

### Build instructions

```bash
    mkdir build
    cmake -S . -B build
    cmake --build build
```

### Testing

To run the test suite from the root directory, use the following command after building:

```bash
    cmake --build build --target test
```

Alternatively, you can run CTest from the build directory

```bash
    cd build
    ctest
```

CTest allows for multiple tests to be run in parallel by using the -j flag.
