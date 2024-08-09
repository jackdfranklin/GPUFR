#include "GPUFR/ff_math.cuh"
#include "GPUFR/types.h"

__device__ u32 ff_add(u32 a, u32 b, u32 p){
	u64 sum = (u64)a%p + (u64)b%p;	
	return (u32)(sum%p);
}

__device__ u32 ff_subtract(u32 a, u32 b, u32 p){
	return ff_add(a, p - b%p, p);
}
