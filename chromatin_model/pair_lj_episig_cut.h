/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/episig/cut,PairLJEpisigCut);
// clang-format on
#else

#ifndef LMP_PAIR_EPISIG_LJ_CUT_H
#define LMP_PAIR_EPISIG_LJ_CUT_H

#include "pair.h"

namespace LAMMPS_NS {

class PairLJEpisigCut : public Pair {
 public:
  PairLJEpisigCut(class LAMMPS *);
  ~PairLJEpisigCut() override;
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
  void born_matrix(int, int, int, int, double, double, double, double &, double &) override;
  void *extract(const char *, int &) override;

  void compute_inner() override;
  void compute_middle() override;
  void compute_outer(int, int) override;

 protected:
  double cut_global;
  double **cut;
  double **epsilon, **sigma;
  double **lj1, **lj2, **lj3, **lj4, **offset;
  double *cut_respa;

  // d2 variable to hold the signals of every Bead object:
  double **d2_signals;

  // ten parameters for the 4x4 symmetric matrix B, which is used to compute the effective epsilon:
  double B11, B12, B13, B14;
  double B22, B23, B24;
  double B33, B34;
  double B44;

  // ten flags to indicate whether the corresponding B-matrix element has been set or not:
  int B11_set, B12_set, B13_set, B14_set;
  int B22_set, B23_set, B24_set;
  int B33_set, B34_set;
  int B44_set;

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif
#endif