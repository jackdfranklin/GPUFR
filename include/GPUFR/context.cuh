#pragma once

#include "GPUFR/detokenize.cuh"
#include "GPUFR/precomp.hpp"

class Context 
{
    private:
    int prime_id;
    
    public:

    Context();
    void new_prime();
    u32 get_prime();
    u32 get_prime(int p_id);
    u32 get_root(int ind);
    u32 get_root(int p_id, int r_id);

    std::vector<u32> get_roots();
    std::vector<u32> get_roots(int p_id);
};