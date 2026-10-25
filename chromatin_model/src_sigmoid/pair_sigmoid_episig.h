/* -*- c++ -*- ----------------------------------------------------------
   pair_style sigmoid/episig

   Signal-weighted contact (switching-function) attraction, linear in B:

     E_ij = -( C_IJ + S_ij ) * g(r_ij)         for r_ij < rcut_IJ
     S_ij = sum_{a,b} x_i^a B_ab x_j^b
     g(r) = f(r) - f(rcut)*shift
     form tanh (default):  f(r) = 1/2 [1 + tanh( mu (rc - r) )]
     form exp           :  same function, evaluated as 1/(1 + exp(-2 mu (rc - r)))
     form rational      :  f(r) = 1 / (1 + (r/rc)^(2n)),  n = integer given in the mu slot
                           (no sqrt / transcendental call; slope at rc = -n/(2 rc), so n = mu*rc
                            matches the tanh slope)

   No repulsive core: combine with a WCA term via pair_style hybrid/overlay.
   Extra quantities (compute pair sigmoid/episig -> vector):
     Phi_ab = dE/d(-B_ab) for a<=b (row-major upper triangle), then N_c = sum g.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(sigmoid/episig,PairSigmoidEpisig);
// clang-format on
#else

#ifndef LMP_PAIR_SIGMOID_EPISIG_H
#define LMP_PAIR_SIGMOID_EPISIG_H

#include "pair.h"

namespace LAMMPS_NS {

class PairSigmoidEpisig : public Pair {
 public:
  static constexpr int MAXSIG = 8;

  PairSigmoidEpisig(class LAMMPS *);
  ~PairSigmoidEpisig() override;

  void compute(int, int) override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  void write_restart_settings(FILE *) override;
  void read_restart_settings(FILE *) override;
  void write_data(FILE *) override;
  void write_data_all(FILE *) override;
  double single(int, int, int, int, double, double, double, double &) override;
  void *extract(const char *, int &) override;

 protected:
  enum { TANH = 0, EXPF = 1, RATIONAL = 2 };
  int form;                        // switching-function form
  int nsig;                        // number of signals per bead
  double cut_global;
  double **cut, **c0, **rc, **mu;  // per type pair
  double **offset;                 // f(rcut) if pair_modify shift yes, else 0
  double **rcinv2;                 // 1/rc^2 (rational form)
  int **nexp;                      // integer exponent n (rational form)
  double Bmat[MAXSIG][MAXSIG];
  int Bset[MAXSIG][MAXSIG];

  virtual void allocate();
  template <int FORM, int NS> void eval(int, int);
  double switch_f(int, int, double) const;
  void set_nsig(int);
  double **get_signals();
  inline int tri_index(int a, int b) const { return a * nsig - a * (a - 1) / 2 + (b - a); }
};

}    // namespace LAMMPS_NS

#endif
#endif
