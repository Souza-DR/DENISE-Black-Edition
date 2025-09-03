/*------------------------------------------------------------------------
 * Minimal SPG utilities: compute BB (Barzilai–Borwein) spectral step and
 * store previous model/gradient tiles for next iteration. Designed to be
 * non-intrusive and reuse DENISE infrastructure.
 * ----------------------------------------------------------------------*/

#include "fd.h"
#include <math.h>

/* Build local tile filenames under JACOBIAN for SPG. */
static void spg_fname(char *buf, const char *what, int r1, int r2){
  /* e.g., <JACOBIAN>_spg_prev_vp.<POS1>.<POS2> */
  extern char JACOBIAN[STRING_SIZE];
  sprintf(buf, "%s_spg_%s.%i.%i", JACOBIAN, what, r1, r2);
}

/* Internal: BB epsilon from two fields (model s, gradient y). Reduces globally. */
static float spg_bb_eps_generic(float **m_now, float **m_prev, float **g_now, float **g_prev){
  extern int NX, NY, IDX, IDY;
  double num_local = 0.0, den_local = 0.0;
  for (int i=1;i<=NX;i+=IDX){
    for (int j=1;j<=NY;j+=IDY){
      const double s = (double)m_now[j][i] - (double)m_prev[j][i];
      const double y = (double)g_now[j][i] - (double)g_prev[j][i];
      num_local += s*y;
      den_local += y*y;
    }
  }
  double num=0.0, den=0.0;
  MPI_Allreduce(&num_local, &num, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&den_local, &den, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const double eps = (den>0.0) ? num/den : 0.0;
  /* Clamp to safe range to avoid pathologies; fall back handled by caller. */
  const double eps_min = 1e-8, eps_max = 1e3;
  return (float)fmin(fmax(eps, eps_min), eps_max);
}

float spg_bb_eps_psv(float **ppi, float **pu, float **prho,
                     float **g_vp, float **g_vs, float **g_rho){
  extern int POS[3];
  float eps = 0.0f;

  /* Try to read previous tiles; if any missing, return 0 to signal fallback. */
  char f_mvp[STRING_SIZE], f_mvs[STRING_SIZE], f_mrho[STRING_SIZE];
  char f_gvp[STRING_SIZE], f_gvs[STRING_SIZE], f_grho[STRING_SIZE];
  spg_fname(f_mvp,  "prev_vp",  POS[1], POS[2]);
  spg_fname(f_mvs,  "prev_vs",  POS[1], POS[2]);
  spg_fname(f_mrho, "prev_rho", POS[1], POS[2]);
  spg_fname(f_gvp,  "prev_gvp", POS[1], POS[2]);
  spg_fname(f_gvs,  "prev_gvs", POS[1], POS[2]);
  spg_fname(f_grho, "prev_grho",POS[1], POS[2]);

  FILE *Fmvp=fopen(f_mvp,"rb"), *Fmvs=fopen(f_mvs,"rb"), *Fmr=fopen(f_mrho,"rb");
  FILE *Fgvp=fopen(f_gvp,"rb"), *Fgvs=fopen(f_gvs,"rb"), *Fgr=fopen(f_grho,"rb");
  if(!(Fmvp&&Fmvs&&Fmr&&Fgvp&&Fgvs&&Fgr)){
    if(Fmvp) fclose(Fmvp); if(Fmvs) fclose(Fmvs); if(Fmr) fclose(Fmr);
    if(Fgvp) fclose(Fgvp); if(Fgvs) fclose(Fgvs); if(Fgr) fclose(Fgr);
    return 0.0f; /* fallback to configured EPS_SCALE */
  }

  extern int NX, NY, IDX, IDY;
  /* Read prev tiles into temporary matrices with same shape. */
  float **mvp_prev = matrix(1,NY,1,NX), **mvs_prev = matrix(1,NY,1,NX), **mrho_prev = matrix(1,NY,1,NX);
  float **gvp_prev = matrix(1,NY,1,NX), **gvs_prev = matrix(1,NY,1,NX), **grho_prev = matrix(1,NY,1,NX);

  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fread(&mvp_prev[j][i], sizeof(float),1,Fmvp); }}
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fread(&mvs_prev[j][i], sizeof(float),1,Fmvs); }}
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fread(&mrho_prev[j][i],sizeof(float),1,Fmr ); }}
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fread(&gvp_prev[j][i], sizeof(float),1,Fgvp); }}
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fread(&gvs_prev[j][i], sizeof(float),1,Fgvs); }}
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fread(&grho_prev[j][i],sizeof(float),1,Fgr ); }}

  fclose(Fmvp); fclose(Fmvs); fclose(Fmr);
  fclose(Fgvp); fclose(Fgvs); fclose(Fgr);

  float eps_vp  = spg_bb_eps_generic(ppi,  mvp_prev,  g_vp,  gvp_prev);
  float eps_vs  = spg_bb_eps_generic(pu,   mvs_prev,  g_vs,  gvs_prev);
  float eps_rho = spg_bb_eps_generic(prho, mrho_prev, g_rho, grho_prev);

  /* Simple average of positive candidates; ignore zeros. */
  int cnt=0; double acc=0.0;
  if(eps_vp  > 0.0f){ acc+=eps_vp;  cnt++; }
  if(eps_vs  > 0.0f){ acc+=eps_vs;  cnt++; }
  if(eps_rho > 0.0f){ acc+=eps_rho; cnt++; }
  eps = (cnt>0)?(float)(acc/cnt):0.0f;

  free_matrix(mvp_prev,1,NY,1,NX); free_matrix(mvs_prev,1,NY,1,NX); free_matrix(mrho_prev,1,NY,1,NX);
  free_matrix(gvp_prev,1,NY,1,NX); free_matrix(gvs_prev,1,NY,1,NX); free_matrix(grho_prev,1,NY,1,NX);
  return eps;
}

void spg_store_prev_psv(float **ppi, float **pu, float **prho,
                        float **g_vp, float **g_vs, float **g_rho){
  extern int POS[3], NX, NY, IDX, IDY;
  char f_mvp[STRING_SIZE], f_mvs[STRING_SIZE], f_mrho[STRING_SIZE];
  char f_gvp[STRING_SIZE], f_gvs[STRING_SIZE], f_grho[STRING_SIZE];
  spg_fname(f_mvp,  "prev_vp",  POS[1], POS[2]);
  spg_fname(f_mvs,  "prev_vs",  POS[1], POS[2]);
  spg_fname(f_mrho, "prev_rho", POS[1], POS[2]);
  spg_fname(f_gvp,  "prev_gvp", POS[1], POS[2]);
  spg_fname(f_gvs,  "prev_gvs", POS[1], POS[2]);
  spg_fname(f_grho, "prev_grho",POS[1], POS[2]);
  FILE *Fmvp=fopen(f_mvp,"wb"), *Fmvs=fopen(f_mvs,"wb"), *Fmr=fopen(f_mrho,"wb");
  FILE *Fgvp=fopen(f_gvp,"wb"), *Fgvs=fopen(f_gvs,"wb"), *Fgr=fopen(f_grho,"wb");
  if(!(Fmvp&&Fmvs&&Fmr&&Fgvp&&Fgvs&&Fgr)){
    if(Fmvp) fclose(Fmvp); if(Fmvs) fclose(Fmvs); if(Fmr) fclose(Fmr);
    if(Fgvp) fclose(Fgvp); if(Fgvs) fclose(Fgvs); if(Fgr) fclose(Fgr);
    return;
  }
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fwrite(&ppi[j][i], sizeof(float),1,Fmvp); }}
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fwrite(&pu[j][i],  sizeof(float),1,Fmvs); }}
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fwrite(&prho[j][i],sizeof(float),1,Fmr ); }}
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fwrite(&g_vp[j][i], sizeof(float),1,Fgvp); }}
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fwrite(&g_vs[j][i], sizeof(float),1,Fgvs); }}
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fwrite(&g_rho[j][i],sizeof(float),1,Fgr ); }}
  fclose(Fmvp); fclose(Fmvs); fclose(Fmr);
  fclose(Fgvp); fclose(Fgvs); fclose(Fgr);
}

float spg_bb_eps_ac(float **ppi, float **prho,
                    float **g_vp, float **g_rho){
  extern int POS[3];
  float eps = 0.0f;
  char f_mvp[STRING_SIZE], f_mrho[STRING_SIZE];
  char f_gvp[STRING_SIZE], f_grho[STRING_SIZE];
  spg_fname(f_mvp,  "prev_vp",  POS[1], POS[2]);
  spg_fname(f_mrho, "prev_rho", POS[1], POS[2]);
  spg_fname(f_gvp,  "prev_gvp", POS[1], POS[2]);
  spg_fname(f_grho, "prev_grho",POS[1], POS[2]);
  FILE *Fmvp=fopen(f_mvp,"rb"), *Fmr=fopen(f_mrho,"rb");
  FILE *Fgvp=fopen(f_gvp,"rb"), *Fgr=fopen(f_grho,"rb");
  if(!(Fmvp&&Fmr&&Fgvp&&Fgr)){
    if(Fmvp) fclose(Fmvp); if(Fmr) fclose(Fmr); if(Fgvp) fclose(Fgvp); if(Fgr) fclose(Fgr);
    return 0.0f;
  }
  extern int NX, NY, IDX, IDY;
  float **mvp_prev = matrix(1,NY,1,NX), **mrho_prev = matrix(1,NY,1,NX);
  float **gvp_prev = matrix(1,NY,1,NX), **grho_prev = matrix(1,NY,1,NX);
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fread(&mvp_prev[j][i], sizeof(float),1,Fmvp); }}
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fread(&mrho_prev[j][i],sizeof(float),1,Fmr ); }}
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fread(&gvp_prev[j][i], sizeof(float),1,Fgvp); }}
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fread(&grho_prev[j][i],sizeof(float),1,Fgr ); }}
  fclose(Fmvp); fclose(Fmr); fclose(Fgvp); fclose(Fgr);
  float eps_vp  = spg_bb_eps_generic(ppi,  mvp_prev,  g_vp,  gvp_prev);
  float eps_rho = spg_bb_eps_generic(prho, mrho_prev, g_rho, grho_prev);
  int cnt=0; double acc=0.0; if(eps_vp>0.0f){acc+=eps_vp;cnt++;} if(eps_rho>0.0f){acc+=eps_rho;cnt++;}
  float out = (cnt>0)?(float)(acc/cnt):0.0f;
  free_matrix(mvp_prev,1,NY,1,NX); free_matrix(mrho_prev,1,NY,1,NX);
  free_matrix(gvp_prev,1,NY,1,NX); free_matrix(grho_prev,1,NY,1,NX);
  return out;
}

void spg_store_prev_ac(float **ppi, float **prho,
                       float **g_vp, float **g_rho){
  extern int POS[3], NX, NY, IDX, IDY;
  char f_mvp[STRING_SIZE], f_mrho[STRING_SIZE];
  char f_gvp[STRING_SIZE], f_grho[STRING_SIZE];
  spg_fname(f_mvp,  "prev_vp",  POS[1], POS[2]);
  spg_fname(f_mrho, "prev_rho", POS[1], POS[2]);
  spg_fname(f_gvp,  "prev_gvp", POS[1], POS[2]);
  spg_fname(f_grho, "prev_grho",POS[1], POS[2]);
  FILE *Fmvp=fopen(f_mvp,"wb"), *Fmr=fopen(f_mrho,"wb");
  FILE *Fgvp=fopen(f_gvp,"wb"), *Fgr=fopen(f_grho,"wb");
  if(!(Fmvp&&Fmr&&Fgvp&&Fgr)){
    if(Fmvp) fclose(Fmvp); if(Fmr) fclose(Fmr); if(Fgvp) fclose(Fgvp); if(Fgr) fclose(Fgr);
    return;
  }
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fwrite(&ppi[j][i], sizeof(float),1,Fmvp); }}
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fwrite(&prho[j][i],sizeof(float),1,Fmr ); }}
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fwrite(&g_vp[j][i], sizeof(float),1,Fgvp); }}
  for(int i=1;i<=NX;i+=IDX){ for(int j=1;j<=NY;j+=IDY){ fwrite(&g_rho[j][i],sizeof(float),1,Fgr ); }}
  fclose(Fmvp); fclose(Fmr); fclose(Fgvp); fclose(Fgr);
}
