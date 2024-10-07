#include "GPUFR/ff_math.cuh"
#include "GPUFR/types.h"

__device__ u32 ff_add(u32 a, u32 b, u32 p){
	u64 sum = (u64)a%p + (u64)b%p;	
	return (u32)(sum%p);
}

__device__ u32 ff_subtract(u32 a, u32 b, u32 p){
	return ff_add(a, p - b%p, p);
}

__device__ u32 ff_multiply(u32 a, u32 b, u32 p){
	u64 prod = (u64)a * (u64)b;	
	return (u32)(prod%(u64)p);
}

__device__ u32 ff_pow(u32 m, u32 exp, u32 p){
	u32 result = m%p;
	if (exp > 0)
	{
		for(int i = 0; i < exp-1; i++){
			result = ff_multiply(result, m, p);
		}
	} else {
		result = 1;
	}
	return result;
}

__device__ u32 ff_divide(u32 a, u32 b, u32 p){
	u32 b_inv = modular_inverse(b, p);
	return ff_multiply(a, b_inv, p);
}

__device__ u32 modular_inverse(u32 a, u32 p){

	u64 r0 = a%p;
	u64 r1 = p;

	i64 s0 = 1;
	i64 s1 = 0;

	while(r1){
		u64 q = r0 / r1;

		u64 r_temp = r0 - q * r1;
		r0 = r1;
		r1 = r_temp;

		i64 s_temp = s0 - q * s1;
		s0 = s1;
		s1 = s_temp;

	}

	return (u32)( (p + s0)%p );

}
