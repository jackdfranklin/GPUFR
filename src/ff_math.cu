#include "GPUFR/ff_math.cuh"
#include "GPUFR/types.h"
#include <stdio.h>

__device__ u32 ff_add(u32 a, u32 b, u32 p){
	u32 sum = a%p + b%p;	
	return sum%p;
}

__device__ u32 ff_subtract(u32 a, u32 b, u32 p){
	u32 sub = a%p - b%p + p;
	return sub%p;
}

// Doesnt seem to work for some numbers e.g. 1000071137*750053377 mod 1000071169
// __host__ __device__ u32 ff_multiply_2(u32 a, u32 b, u32 p){
// 	a = a%p;
// 	b = b%p;
// 	double x = static_cast<double>(a);
// 	u32 c = static_cast<u32>( (x*b) / p );
// 	u32 r = ((a*b) - (c*p)) % p;
// 	return r;
// }

__device__ u32 ff_multiply(u32 a, u32 b, u32 p){
	u64 prod = (u64)(a%p) * (u64)(b%p);
	u32 res = (u32)(prod%(u64)p);
	return res;
}


// Note that conditionals on the exp shouldnt cause warp divergences as the exp will match in every thread
__device__ u32 ff_pow(u32 m, u32 exp, u32 p){
	u32 result = 1;
	if (exp > 0)
	{
		while (exp > 0)
		{
			if (exp%2 == 1)
				result = ff_multiply(result, m, p);
			exp = exp>>1;
			m = ff_multiply(m, m, p);
		}
	}
	return result;
}

__device__ u32 ff_divide(u32 a, u32 b, u32 p){
	u32 b_inv = modular_inverse(b, p);
	return ff_multiply(a, b_inv, p);
}

__device__ u32 modular_inverse(u32 a, u32 p){
	u32 r1, r2, rTmp;
	u32 q;

	i32 t1, t2, tTmp;

	r1 = p;
	r2 = a%p;

	t1 = 0;
	t2 = 1;

	while (r2)
	{
		q = r1/r2;

		rTmp = r2;
		r2 = r1 - q*r2;
		r1 = rTmp;

		tTmp = t2;
		t2 = t1 - q*t2;
		t1 = tTmp;
	}

	return t1+p;
}
