/* -*- c++ -*- ----------------------------------------------------------
   pair_style sigmoid/episig/omp  --  OpenMP version of sigmoid/episig.
   Same syntax, energy, forces and compute-pair vector (Phi_ab, N_c).
   Requires the OPENMP package (place in src/OPENMP/ or src/ with PKG_OPENMP=on).
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(sigmoid/episig/omp,PairSigmoidEpisigOMP);
// clang-format on
#else

#ifndef LMP_PAIR_SIGMOID_EPISIG_OMP_H
#define LMP_PAIR_SIGMOID_EPISIG_OMP_H

#include "pair_sigmoid_episig.h"
#include "thr_omp.h"

namespace LAMMPS_NS {

class PairSigmoidEpisigOMP : public PairSigmoidEpisig, public ThrOMP {

 public:
  PairSigmoidEpisigOMP(class LAMMPS *);

  void compute(int, int) override;
  double memory_usage() override;

 private:
  double **sig_omp;    // signals array for the current call (shared, read-only)

  template <int EVFLAG, int EFLAG, int NEWTON_PAIR, int FORM, int NS>
  void eval(int ifrom, int ito, ThrData *const thr, double *acc);

  template <int FORM, int NS>
  void dispatch(int eflag, int ifrom, int ito, ThrData *const thr, double *acc);
};

}    // namespace LAMMPS_NS

#endif
#endif
