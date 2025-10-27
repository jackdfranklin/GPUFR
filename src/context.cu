#include "GPUFR/context.cuh"

Context::Context()
{
    prime_id = 0;
}

void Context::new_prime()
{
    prime_id += 1;
}

u32 Context::get_prime()
{
    return precomp[prime_id][0];
}

u32 Context::get_root(int ind)
{
    return precomp[prime_id][ind];
}

std::vector<u32> Context::get_roots()
{
    return std::vector<u32>(std::begin(precomp[prime_id]), std::end(precomp[prime_id]));
}
