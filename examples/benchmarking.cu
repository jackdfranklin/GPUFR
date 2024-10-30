#include "GPUFR/lagrange_solver.cuh"

int main(int argc, char* argv[])
{
    std::string ntt_primes = argv[1];
    multi_interp(1, 12, ntt_primes);
}