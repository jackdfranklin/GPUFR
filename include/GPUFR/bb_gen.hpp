#pragma once

#include <vector>
#include <string>

#include <cuda.h>

void gen_bb_module(const std::string &expression, const std::vector<std::string> &vars, CUcontext cuda_context, CUdevice cuda_device, CUmodule &module);

#define NVJITLINK_SAFE_CALL(h,x)                                  \
  do {                                                            \
    nvJitLinkResult result = x;                                   \
    if (result != NVJITLINK_SUCCESS) {                            \
      std::cerr << "\nerror: " #x " failed with error "           \
                << result << '\n';                                \
      size_t lsize;                                               \
      result = nvJitLinkGetErrorLogSize(h, &lsize);               \
      if (result == NVJITLINK_SUCCESS && lsize > 0) {             \
        char *log = (char*)malloc(lsize);                         \
    result = nvJitLinkGetErrorLog(h, log);                        \
    if (result == NVJITLINK_SUCCESS) {                            \
      std::cerr << "error: " << log << '\n';                      \
      free(log);                                                  \
    }                                                             \
      }                                                           \
      exit(1);                                                    \
    }                                                             \
  } while(0)

