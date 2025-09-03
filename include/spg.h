/* Simple SPG (Spectral Projected Gradient) helpers for DENISE */
#ifndef SPG_H
#define SPG_H

/* Compute Barzilai–Borwein spectral step for PSV case using local tiles. */
float spg_bb_eps_psv(float **ppi, float **pu, float **prho,
                     float **g_vp, float **g_vs, float **g_rho);

/* Persist previous model and gradient for PSV (local tiles). */
void spg_store_prev_psv(float **ppi, float **pu, float **prho,
                        float **g_vp, float **g_vs, float **g_rho);

/* ACoustic variant (Vp, Rho). */
float spg_bb_eps_ac(float **ppi, float **prho,
                    float **g_vp, float **g_rho);
void spg_store_prev_ac(float **ppi, float **prho,
                       float **g_vp, float **g_rho);

#endif /* SPG_H */

