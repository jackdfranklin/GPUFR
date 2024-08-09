#pragma once

#include "GPUFR/types.h"

__device__ u32 ff_add(u32 a, u32 b, u32 p);

__device__ u32 ff_subtract(u32 a, u32 b, u32 p);
