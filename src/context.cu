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

u32 Context::get_prime(int p_id)
{
    return precomp[p_id][0];
}

u32 Context::get_root(int ind)
{
    return precomp[prime_id][ind];
}

u32 Context::get_root(int p_id, int r_id)
{
    return precomp[p_id][r_id];
}

std::vector<u32> Context::get_roots()
{
    return std::vector<u32>(std::begin(precomp[prime_id]), std::end(precomp[prime_id]));
}

std::vector<u32> Context::get_roots(int p_id)
{
    return std::vector<u32>(std::begin(precomp[p_id]), std::end(precomp[p_id]));
}
