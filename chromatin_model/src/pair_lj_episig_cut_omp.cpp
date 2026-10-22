/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include "pair_lj_episig_cut_omp.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "respa.h"
#include "update.h"
#include "suffix.h"

#include <cmath>
#include <cstring>
#include "omp_compat.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairLJEpisigCutOMP::PairLJEpisigCutOMP(LAMMPS *lmp) :
  PairLJEpisigCut(lmp), ThrOMP(lmp, THR_PAIR)
{
  suffix_flag |= Suffix::OMP;

  respa_enable = 0;
  born_matrix_enable = 1;
  writedata = 1;

  // initializing signal-related variables:
  d2_signals = nullptr;

}

/* ---------------------------------------------------------------------- */

void PairLJEpisigCutOMP::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;

//----------------------------------------------------------------------------------------------------
// RETRIEVING THE SIGNALS PROPERTY FROM THE ATOM CLASS.
  d2_signals = atom->darray[signals_index];
//----------------------------------------------------------------------------------------------------

// loop over neighbors of my atoms
//----------------------------------------------------------------------------------------------------

  #if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag,vflag)
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);

    ThrData *thr = fix->get_thr(tid);

    thr->timer(Timer::START);

    ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

    if (evflag) {
      if (eflag) {
        if (force->newton_pair)
          eval<1,1,1>(ifrom, ito, thr);
        else
          eval<1,1,0>(ifrom, ito, thr);
      } else {
        if (force->newton_pair)
          eval<1,0,1>(ifrom, ito, thr);
        else
          eval<1,0,0>(ifrom, ito, thr);
      }
    } else {
      if (force->newton_pair)
        eval<0,0,1>(ifrom, ito, thr);
      else
        eval<0,0,0>(ifrom, ito, thr);
    }

    thr->timer(Timer::PAIR);

    reduce_thr(this, eflag, vflag, thr);
  }
}


template <int EVFLAG, int EFLAG, int NEWTON_PAIR>
void PairLJEpisigCutOMP::eval(int ifrom, int ito, ThrData * const thr)
{
  const auto * _noalias const x = (dbl3_t *) atom->x[0];
  auto * _noalias const f = (dbl3_t *) thr->get_f()[0];

  const int * _noalias const type = atom->type;
  const double * _noalias const special_lj = force->special_lj;

  const int * _noalias const ilist = list->ilist;
  const int * _noalias const numneigh = list->numneigh;
  const int * const * const firstneigh = list->firstneigh;

  const int nlocal = atom->nlocal;

  for (int ii = ifrom; ii < ito; ++ii) {

    const int i = ilist[ii];
    const int itype = type[i];

    const int * _noalias const jlist = firstneigh[i];

    const double * _noalias const cutsqi = cutsq[itype];
    const double * _noalias const offseti = offset[itype];
    const double * _noalias const lj1i = lj1[itype];
    const double * _noalias const lj2i = lj2[itype];
    const double * _noalias const lj3i = lj3[itype];
    const double * _noalias const lj4i = lj4[itype];
    const double * _noalias const epsiloni = epsilon[itype];

    const double xtmp = x[i].x;
    const double ytmp = x[i].y;
    const double ztmp = x[i].z;

    double fxtmp = 0.0;
    double fytmp = 0.0;
    double fztmp = 0.0;

    const double *si = d2_signals[i];

    const int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; ++jj) {

      const int j = jlist[jj] & NEIGHMASK;
      const int jtype = type[j];

      const double factor_lj = special_lj[sbmask(jlist[jj])];

      const double *sj = d2_signals[j];

      // ------------------------------------------------------------
      // Signal-dependent interaction
      // Sij = si^T B sj
      // ------------------------------------------------------------

      double Sij = 0.0;

      for (int alpha = 0; alpha < nsig; ++alpha) {
        for (int beta = 0; beta < nsig; ++beta) {
          Sij += si[alpha] * B[alpha][beta] * sj[beta];
        }
      }

      const double epsilon0 = epsiloni[jtype];
      const double epsilon_eff = std::max(epsilon0, Sij);
      const double eps_scale = epsilon_eff / epsilon0;

      if (!std::isfinite(eps_scale)) {
        error->all(FLERR, "Non-finite epsilon scaling encountered");
      }

      // ------------------------------------------------------------
      // LJ interaction
      // ------------------------------------------------------------

      const double delx = xtmp - x[j].x;
      const double dely = ytmp - x[j].y;
      const double delz = ztmp - x[j].z;

      const double rsq = delx * delx + dely * dely + delz * delz;

      if (rsq < cutsqi[jtype]) {

        const double r2inv = 1.0 / rsq;
        const double r6inv = r2inv * r2inv * r2inv;

        const double forcelj =
          r6inv *
          (lj1i[jtype] * eps_scale * r6inv
           - lj2i[jtype] * eps_scale);

        const double fpair =
          factor_lj * forcelj * r2inv;

        // Accumulate force on i locally.
        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        // Force on j goes into the thread-local force array.
        if (NEWTON_PAIR || j < nlocal) {
          f[j].x -= delx * fpair;
          f[j].y -= dely * fpair;
          f[j].z -= delz * fpair;
        }

        // ----------------------------------------------------------
        // Energy
        // ----------------------------------------------------------

        double evdwl = 0.0;

        if (EFLAG) {
          evdwl =
            r6inv *
            (lj3i[jtype] * eps_scale * r6inv
             - lj4i[jtype] * eps_scale)
            - offseti[jtype];

          evdwl *= factor_lj;
        }

        // ----------------------------------------------------------
        // Energy / virial bookkeeping
        // ----------------------------------------------------------

        if (EVFLAG) {
          ev_tally_thr(
            this,
            i,
            j,
            nlocal,
            NEWTON_PAIR,
            evdwl,
            0.0,
            fpair,
            delx,
            dely,
            delz,
            thr
          );
        }
      }
    }

    // Add accumulated force on i once.
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
  }
}