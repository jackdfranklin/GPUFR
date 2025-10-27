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
    u32 get_root(int ind);
    std::vector<u32> get_roots();
};