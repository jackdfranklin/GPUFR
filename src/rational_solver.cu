//TODO
/*
1. We intend to interpolate functions of the form F(xi) = N(xi) / D(xi) where N and D are polynomials with integer coefficients.
2. This can be done by interpolating univariate rational functions over a grid of probes as with the Lagrange interpolation of polynomials.
3. Univariate rational interpolation can be done using a modified version of the extended Euclidean algorithm.
4. The algorithm is as follows:
    a. We wish to solve for Q(x) and P(x) such that R(x) Q(x) - P(x) = 0.
    b. We begin by determining the polynomial R(x) by lagrange interpolation over the probes within a chosen Galois field.
*/

#include "GPUFR/rational_solver.cuh"

void solve_pade(interp_data id) {
    
}