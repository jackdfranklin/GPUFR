#include <catch2/catch_test_macros.hpp>
#include <random>
#include <stdio.h>

#include "GPUFR/crt.cuh"
#include <gmp.h>

std::random_device rd; // Seed
std::mt19937 gen(rd()); // Mersenne Twister engine
std::uniform_int_distribution<u64> dist(1000071169/2, 1000071169); // Range [1, 100]

TEST_CASE("CRT Test"){
    // Generate 2 arrays of u64 numbers
    // Take modulo wrt 2 differernt moduli
    // Generate 3rd output arr
    // Pass all to crt
    // Check input arr matches output

    int arr_size = 1;
    u32 primes[] = {1000071169,
                    1000112129};
    
    int n_moduli = 2;


    // u64* base_arr = new u64[arr_size];
    u64 base_arr[] = {500035584};
    u64* out_arr = new u64[arr_size];
    u32** mod_arrs = new u32*[n_moduli];

    for (int i=0; i<n_moduli; i++)
        mod_arrs[i] = new u32[arr_size];

    for (int i=0; i<arr_size; i++)
    {
        // base_arr[i] = dist(gen);
        for (int j=0; j<n_moduli; j++)
            mod_arrs[j][i] = base_arr[i]%primes[j];
    }

    compute_crt(mod_arrs, primes, out_arr, arr_size, n_moduli);

    for (int i=0; i<arr_size; i++)
    {
        REQUIRE(out_arr[i] == base_arr[i]); 
    }
}

TEST_CASE("CRT Small Test"){
    // Generate 2 arrays of u64 numbers
    // Take modulo wrt 2 differernt moduli
    // Generate 3rd output arr
    // Pass all to crt
    // Check input arr matches output

    int arr_size = 1;
    int n_moduli = 3;

    u32 primes[] = {3, 5, 7};

    u64 base_arr[] = {23};
    u64* out_arr = new u64[arr_size];
    u32** mod_arrs = new u32*[n_moduli];

    for (int i=0; i<n_moduli; i++)
        mod_arrs[i] = new u32[arr_size];

    for (int i=0; i<arr_size; i++)
    {
        for (int j=0; j<n_moduli; j++)
            mod_arrs[j][i] = base_arr[i]%primes[j];
    }
    
    compute_crt(mod_arrs, primes, out_arr, arr_size, n_moduli);

    for (int i=0; i<arr_size; i++)
    {
        REQUIRE(out_arr[i] == base_arr[i]); 
    }
}

TEST_CASE("CRT GMP Test"){
    int arr_size = 1;
    u32 primes[] = {1000112129,
                    1000210433,
                    1000308737,
                    1000800257};
    
    int n_moduli = 4;

    mpz_t base_arr[arr_size];
    mpz_init(base_arr[0]);
    mpz_set_str(base_arr[0], "500035584500035584500035584", 10);
    mpz_t* out_arr = new mpz_t[arr_size];
    u32** mod_arrs = new u32*[n_moduli];

    for (int i=0; i<n_moduli; i++)
        mod_arrs[i] = new u32[arr_size];
    
    mpz_t tmp;
    mpz_init(tmp);

    for (int i=0; i<arr_size; i++)
    {
        for (int j=0; j<n_moduli; j++)
        {
            mpz_mod_ui(tmp, base_arr[i], primes[j]);
            mod_arrs[j][i] = mpz_get_ui(tmp);
        }
    }

    compute_crt(mod_arrs, primes, out_arr, arr_size, n_moduli);

    for (int i=0; i<arr_size; i++)
    {
        REQUIRE(mpz_cmp(out_arr[i], base_arr[i]) == 0); 
    }

    mpz_clear(tmp);
    mpz_clear(base_arr[0]);
    mpz_clear(out_arr[0]);
}

TEST_CASE("CRT GMP u32 Test"){
    int arr_size = 1;

    std::string big_num = "10001121281000112128100011212";

    mpz_t big_mpz;
    mpz_init(big_mpz);
    mpz_set_str(big_mpz, big_num.c_str(), 10);

    mpz_t prime1;
    mpz_init(prime1);
    mpz_set_ui(prime1, 18446744073709551557);
    mpz_t arr1[arr_size];
    mpz_init(arr1[0]);
    mpz_set_str(arr1[0], big_num.c_str(), 10);
    mpz_mod(arr1[0], arr1[0], prime1);

    u32 prime2 = 1000112129;
    mpz_t tmp;
    mpz_init(tmp);
    mpz_set_str(tmp, big_num.c_str(), 10);
    mpz_mod_ui(tmp, tmp, prime2);
    u32 arr2[arr_size];
    arr2[0] = mpz_get_ui(tmp);

    mpz_t out_arr[arr_size];
    mpz_init(out_arr[0]);

    compute_crt(out_arr, arr1, arr2, prime1, prime2, arr_size);

    for (int i=0; i<arr_size; i++)
    {
        REQUIRE(mpz_cmp(out_arr[i], big_mpz) == 0); 
    }

    mpz_clear(tmp);
    mpz_clear(big_mpz);
    mpz_clear(prime1);
    mpz_clear(tmp);
    mpz_clear(arr1[0]);
    mpz_clear(out_arr[0]);
}