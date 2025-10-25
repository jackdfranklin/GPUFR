#include "flint/fmpz_mod.h"
#include "flint/fmpz_mod_poly.h"
#include <array>

int poly_coeffs[] = {1, -1, 1, -1};
unsigned int p = 7;

void EEA(const fmpz_mod_poly_t& A, const fmpz_mod_poly_t& B, fmpz_mod_poly_t&P, fmpz_mod_poly_t& Q, const fmpz_mod_ctx_t& ctx) {

    fmpz_mod_poly_t A_tmp, B_tmp, R_tmp, Q_tmp, vk, vkp1, vk_tmp, vk_tmp2;
    fmpz_mod_poly_init(A_tmp, ctx);
    fmpz_mod_poly_init(B_tmp, ctx);
    fmpz_mod_poly_init(R_tmp, ctx);
    fmpz_mod_poly_init(Q_tmp, ctx);
    fmpz_mod_poly_init(vk, ctx);
    fmpz_mod_poly_init(vkp1, ctx);
    fmpz_mod_poly_init(vk_tmp, ctx);
    fmpz_mod_poly_init(vk_tmp2, ctx);

    fmpz_mod_poly_set_coeff_ui(vk, 0, 0, ctx);
    fmpz_mod_poly_set_coeff_ui(vkp1, 0, 1, ctx);

    fmpz_mod_poly_set(A_tmp, A, ctx);
    fmpz_mod_poly_set(B_tmp, B, ctx);

    bool continue_cond = true;
    slong degree_vkp1;
    while (continue_cond) {
        fmpz_mod_poly_divrem(Q_tmp, R_tmp, A_tmp, B_tmp, ctx); // Get Q and R

        flint_printf("Q = %{fmpz_mod_poly} (%{fmpz_mod_ctx})\n", Q_tmp, ctx);
        flint_printf("R = %{fmpz_mod_poly} (%{fmpz_mod_ctx})\n", R_tmp, ctx);

        // v_k+1 = v_k-1 - Q_k v_k
        fmpz_mod_poly_mul(vk_tmp, vkp1, Q_tmp, ctx);
        fmpz_mod_poly_scalar_mul_ui(vk_tmp2, vk_tmp, p-1, ctx);
        fmpz_mod_poly_set(vk_tmp, vkp1, ctx);
        fmpz_mod_poly_add(vkp1, vk, vk_tmp2, ctx);
        fmpz_mod_poly_set(vk, vk_tmp, ctx);

        flint_printf("vkp = %{fmpz_mod_poly} (%{fmpz_mod_ctx})\n", vk, ctx);
        flint_printf("vkp1 = %{fmpz_mod_poly} (%{fmpz_mod_ctx})\n", vkp1, ctx);

        fmpz_mod_poly_set(A_tmp, B_tmp, ctx);
        fmpz_mod_poly_set(B_tmp, R_tmp, ctx);

        degree_vkp1 = fmpz_mod_poly_degree(vkp1, ctx);
        flint_printf("degree %{slong} \n", degree_vkp1);
        continue_cond = (degree_vkp1 <= 1);
    }

    fmpz_mod_poly_set(P, A_tmp, ctx);
    fmpz_mod_poly_set(Q, vk, ctx);
}

void pade(const fmpz_mod_poly_t& Tmn, fmpz_mod_poly_t& P, fmpz_mod_poly_t& Q, const fmpz_mod_ctx_t& ctx)
{
    fmpz_mod_poly_t xmn1, q, r;
    fmpz_mod_poly_init(xmn1, ctx);
    fmpz_mod_poly_init(q, ctx);
    fmpz_mod_poly_init(r, ctx);

    slong degree = fmpz_mod_poly_degree(Tmn, ctx);
    fmpz_mod_poly_set_coeff_ui(xmn1, degree+1, 1, ctx);

    fmpz_mod_poly_divrem(q, r, xmn1, Tmn, ctx);

    // flint_printf("xmn1 = %{fmpz_mod_poly} (%{fmpz_mod_ctx})\n", xmn1, ctx);
    // flint_printf("q = %{fmpz_mod_poly} (%{fmpz_mod_ctx})\n", q, ctx);
    // flint_printf("r = %{fmpz_mod_poly} (%{fmpz_mod_ctx})\n", r, ctx);

    EEA(xmn1, Tmn, P, Q, ctx);

    flint_printf("P = %{fmpz_mod_poly} (%{fmpz_mod_ctx})\n", P, ctx);
    flint_printf("Q = %{fmpz_mod_poly} (%{fmpz_mod_ctx})\n", Q, ctx);

}

int main(void)
{
    fmpz_mod_ctx_t ctx;
    fmpz_mod_poly_t x, y, P, Q;

    fmpz_mod_ctx_init_ui(ctx, p);

    fmpz_mod_poly_init(x, ctx);
    fmpz_mod_poly_init(y, ctx);
    fmpz_mod_poly_init(P, ctx);
    fmpz_mod_poly_init(Q, ctx);

    for (int i=0; i<4; i++){
        unsigned int coeff;
        coeff = poly_coeffs[i];
        if (poly_coeffs[i] < 0){
            coeff = p+poly_coeffs[i];
        }
        fmpz_mod_poly_set_coeff_ui(x, i, coeff, ctx);
    }

    flint_printf("Tmn = %{fmpz_mod_poly} (%{fmpz_mod_ctx})\n",
                 x, ctx);
    
    pade(x, P, Q, ctx);

    fmpz_mod_poly_clear(x, ctx);
    fmpz_mod_ctx_clear(ctx);
}