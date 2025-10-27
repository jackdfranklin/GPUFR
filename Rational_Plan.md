# Rational Plan

The plan is to impliment a multivariate Padé solver. Starting with the 1d case, that is to solve the equation:

Q(x)S(x) - P(x) = 0 mod x^(2N)

for P and Q where S(x) = a0 + a1 x^1 + ... an x^(2N-1). S(x) is the truncated interpolating polynomial to the rational R(x).

## Finding S(x)

This is equivelent to finding the Taylor series. Working over Galois fields GF(p), this can be computed in O(nlog(n)) time using inverse NTTs. Provided the sample points (xi) are roots of unity (precompute) of p, the coefficients ai are given by:

{ai} = inverse_NTT({R(xi)})

## Solving the Padé equations

P(x) and Q(x) can be found using the Extended Euclidian Algorithm (EEA) applied to S(x) and x^(2N), stopping when the degree of the highest tems in sequence v0 = 0, v1 = 1 vk+1 = vk-1 - qk vk is less than or equal to the degree of the denominator in the rational R(x).

The EEA can be most efficiently implimented using the Half GCD algorthim.

## Chinese Remainder

As with polynomial interpolation, the rationals obtained for each prime must be combined using the CRT

# Steps

0. Determine dimension of P and Q

1. (GPU) Execute Black Box Probes ✅
2. (GPU) Compute {ai} 
    - (GPU) Inverse NTT on probes array
3. (CPU orchestrates) Padé
    - (CPU) Half GCD
    - (GPU) Poly mult
4. (CPU) CRT ✅

## Pseude code

We will use a constant size for all polynomials of
> pol_size = pow(2, ceil(log2(2*N)))

### Main:
> degree of P = dimP - 1

> degree of Q = dimQ - 1

> N = max(dimP, dimQ)
```c++
void interpolate_rational(R, dimP, dimQ)
{
    context ctx;
    N = max(dimP + dimQ);
    pol_size = pow(2, ceil(log2(2*N)))
    device u32* probes; // Device array size 2^(ceil(log2(2N)))
    ctx.new_prime(); // Read in the first prime and its primative roots

    bool probes_successfull = false;
    while (!probes_successfull)
    {
        initialise_roots_of_unity(ctx); // Populate with roots of unity on GPU
        execute_bb(probes, probes_successfull, ctx); // Gets bb results (needs to check valid i.e. no poles)
        if (!probes_successfull)
            ctx.new_prime();
    }

    inverse_ntt(probes); // Compute the ak and store in probes

    device u32* P;
    device u32* Q;

    solve_pade(probes, P, Q, dimP, dimQ); // We begin by running on CPU with current implimentation
}
```

### EEA:
```c++
void solve_pade(ak, P, Q, dimP, dimQ)
{

}
```

### Other functions
```c++
void poly_divrem(A, B, P, R);

void poly_add(A, B, O);

void poly_mult(A, B, O);

void poly_scalar_mult(a, P, O);
```

# Plan

Impliment NTT interpolation on GPU, copy back to CPU. Interpolate Pade with FLINT. Write test case for this using black box evals. 

Benchmark with real rational. If necessary impliment the FLINT portion using GPU accelerated divmod (half GCD) and poly mult.