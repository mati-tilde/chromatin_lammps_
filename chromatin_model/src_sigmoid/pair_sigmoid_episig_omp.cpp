/* ----------------------------------------------------------------------
   pair_style sigmoid/episig/omp  --  OpenMP version of sigmoid/episig.
   Structure follows LAMMPS' pair_lj_cut_omp.cpp (OPENMP package).

   E_ij = -(C_IJ + x_i^T B x_j) g(r_ij),  g = f(r) - f(rcut)*shift
   form tanh | exp | rational, as in pair_sigmoid_episig.cpp.

   compute pair sigmoid/episig/omp returns Phi_ab (a<=b) and N_c = sum g.
   Each thread accumulates a private copy; copies are summed at the end of
   the parallel region (no atomics inside the pair loop).
------------------------------------------------------------------------- */

#include "pair_sigmoid_episig_omp.h"

#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neigh_list.h"
#include "suffix.h"

#include <cmath>

#include "omp_compat.h"
using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairSigmoidEpisigOMP::PairSigmoidEpisigOMP(LAMMPS *lmp) :
    PairSigmoidEpisig(lmp), ThrOMP(lmp, THR_PAIR), sig_omp(nullptr)
{
  suffix_flag |= Suffix::OMP;
  respa_enable = 0;
}

/* ---------------------------------------------------------------------- */

void PairSigmoidEpisigOMP::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;

  // re-fetch every call: the per-atom array may be reallocated when atoms migrate
  sig_omp = get_signals();

  // compute pair reads pvector only on steps with global energy tally
  const int accumulate = eflag_global;
  if (accumulate)
    for (int k = 0; k < nextra; k++) pvector[k] = 0.0;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag, vflag)
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    thr->timer(Timer::START);
    ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

    double acc[MAXSIG * (MAXSIG + 1) / 2 + 1];
    for (int k = 0; k < nextra; k++) acc[k] = 0.0;

    if (nsig == 3) {
      if (form == RATIONAL) dispatch<RATIONAL, 3>(eflag, ifrom, ito, thr, acc);
      else if (form == EXPF) dispatch<EXPF, 3>(eflag, ifrom, ito, thr, acc);
      else dispatch<TANH, 3>(eflag, ifrom, ito, thr, acc);
    } else {
      if (form == RATIONAL) dispatch<RATIONAL, 0>(eflag, ifrom, ito, thr, acc);
      else if (form == EXPF) dispatch<EXPF, 0>(eflag, ifrom, ito, thr, acc);
      else dispatch<TANH, 0>(eflag, ifrom, ito, thr, acc);
    }

    if (eflag_global) {
#if defined(_OPENMP)
#pragma omp critical(sigmoid_episig_pvector)
#endif
      for (int k = 0; k < nextra; k++) pvector[k] += acc[k];
    }

    thr->timer(Timer::PAIR);
    reduce_thr(this, eflag, vflag, thr);
  }    // end of omp parallel region
}

/* ---------------------------------------------------------------------- */

template <int FORM, int NS>
void PairSigmoidEpisigOMP::dispatch(int eflag, int ifrom, int ito, ThrData *const thr,
                                    double *acc)
{
  if (evflag) {
    if (eflag) {
      if (force->newton_pair) eval<1, 1, 1, FORM, NS>(ifrom, ito, thr, acc);
      else eval<1, 1, 0, FORM, NS>(ifrom, ito, thr, acc);
    } else {
      if (force->newton_pair) eval<1, 0, 1, FORM, NS>(ifrom, ito, thr, acc);
      else eval<1, 0, 0, FORM, NS>(ifrom, ito, thr, acc);
    }
  } else {
    if (force->newton_pair) eval<0, 0, 1, FORM, NS>(ifrom, ito, thr, acc);
    else eval<0, 0, 0, FORM, NS>(ifrom, ito, thr, acc);
  }
}

/* ---------------------------------------------------------------------- */

template <int EVFLAG, int EFLAG, int NEWTON_PAIR, int FORM, int NS>
void PairSigmoidEpisigOMP::eval(int iifrom, int iito, ThrData *const thr, double *acc)
{
  const auto *_noalias const x = (dbl3_t *) atom->x[0];
  auto *_noalias const f = (dbl3_t *) thr->get_f()[0];
  const int *_noalias const type = atom->type;
  const double *_noalias const special_lj = force->special_lj;
  const int *_noalias const ilist = list->ilist;
  const int *_noalias const numneigh = list->numneigh;
  const int *const *const firstneigh = list->firstneigh;
  double **const sig = sig_omp;

  const int nlocal = atom->nlocal;
  const int ns = NS ? NS : nsig;
  const int ntri = ns * (ns + 1) / 2;
  const bool accumulate = EFLAG && eflag_global;

  // thread-local copy of B: fixed bounds when NS > 0 (unrolled), no aliasing with f
  double Bl[MAXSIG][MAXSIG];
  for (int a = 0; a < ns; a++)
    for (int b = 0; b < ns; b++) Bl[a][b] = Bmat[a][b];

  double evdwl = 0.0;

  for (int ii = iifrom; ii < iito; ++ii) {
    const int i = ilist[ii];
    const int itype = type[i];
    const int *_noalias const jlist = firstneigh[i];
    const double *_noalias const cutsqi = cutsq[itype];
    const double *_noalias const c0i = c0[itype];
    const double *_noalias const rci = rc[itype];
    const double *_noalias const mui = mu[itype];
    const double *_noalias const offseti = offset[itype];
    const double *_noalias const rcinv2i = rcinv2[itype];
    const int *_noalias const nexpi = nexp[itype];
    const double *const si = sig[i];

    const double xtmp = x[i].x;
    const double ytmp = x[i].y;
    const double ztmp = x[i].z;
    const int jnum = numneigh[i];
    double fxtmp = 0.0, fytmp = 0.0, fztmp = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj];
      const double factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      const double delx = xtmp - x[j].x;
      const double dely = ytmp - x[j].y;
      const double delz = ztmp - x[j].z;
      const double rsq = delx * delx + dely * dely + delz * delz;
      const int jtype = type[j];
      if (rsq >= cutsqi[jtype]) continue;

      // S_ij = x_i^T B x_j
      const double *const sj = sig[j];
      double Sij = 0.0;
      for (int a = 0; a < ns; a++) {
        double s = 0.0;
        for (int b = 0; b < ns; b++) s += Bl[a][b] * sj[b];
        Sij += si[a] * s;
      }
      const double amp = c0i[jtype] + Sij;

      double g, fpair;
      if (FORM == RATIONAL) {
        const double u = rsq * rcinv2i[jtype];
        const int n = nexpi[jtype];
        double un1 = 1.0;
        for (int p = 1; p < n; p++) un1 *= u;
        const double den = 1.0 / (1.0 + un1 * u);
        g = den - offseti[jtype];
        fpair = -factor_lj * amp * 2.0 * n * un1 * rcinv2i[jtype] * den * den;
      } else {
        const double r = sqrt(rsq);
        double th;
        if (FORM == EXPF) {
          const double e = exp(-2.0 * mui[jtype] * (rci[jtype] - r));
          th = (1.0 - e) / (1.0 + e);
        } else {
          th = tanh(mui[jtype] * (rci[jtype] - r));
        }
        g = 0.5 * (1.0 + th) - offseti[jtype];
        fpair = -factor_lj * amp * 0.5 * mui[jtype] * (1.0 - th * th) / r;
      }

      fxtmp += delx * fpair;
      fytmp += dely * fpair;
      fztmp += delz * fpair;
      if (NEWTON_PAIR || j < nlocal) {
        f[j].x -= delx * fpair;
        f[j].y -= dely * fpair;
        f[j].z -= delz * fpair;
      }

      if (EFLAG) evdwl = -factor_lj * amp * g;

      if (accumulate) {
        // pair owned once with newton on; split between ranks with newton off
        const double w = factor_lj * g * ((NEWTON_PAIR || j < nlocal) ? 1.0 : 0.5);
        for (int a = 0; a < ns; a++) {
          acc[tri_index(a, a)] += w * si[a] * sj[a];
          for (int b = a + 1; b < ns; b++) acc[tri_index(a, b)] += w * (si[a] * sj[b] + si[b] * sj[a]);
        }
        acc[ntri] += w;
      }

      if (EVFLAG)
        ev_tally_thr(this, i, j, nlocal, NEWTON_PAIR, evdwl, 0.0, fpair, delx, dely, delz, thr);
    }
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;
  }
}

/* ---------------------------------------------------------------------- */

double PairSigmoidEpisigOMP::memory_usage()
{
  double bytes = memory_usage_thr();
  bytes += PairSigmoidEpisig::memory_usage();
  return bytes;
}
