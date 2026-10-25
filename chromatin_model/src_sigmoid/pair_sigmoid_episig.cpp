/* ----------------------------------------------------------------------
   pair_style sigmoid/episig  --  see pair_sigmoid_episig.h for the model.

   Syntax:
     pair_style sigmoid/episig  Nsig  rcut_global  [form tanh|exp|rational]
     pair_coeff I J  C  rc  mu  [rcut]       # baseline C, switch centre rc, steepness mu
                                             # (form rational: mu slot = integer n)
     pair_coeff * * B a b value              # B_ab (a,b = 1..Nsig), symmetric
       (under hybrid/overlay: pair_coeff * * sigmoid/episig B a b value)
     pair_modify shift yes                   # use g(r) = f(r) - f(rcut): continuous energy

   Sign convention: positive C + S_ij  ->  attraction of depth (C + S_ij).
   No max(eps0, .) clamp: E is exactly linear in B and C, as required by max-ent.
------------------------------------------------------------------------- */

#include "pair_sigmoid_episig.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairSigmoidEpisig::PairSigmoidEpisig(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 1;
  nsig = 0;
  form = TANH;
  nextra = 0;
  pvector = nullptr;
  cut_global = 0.0;
  for (int a = 0; a < MAXSIG; a++)
    for (int b = 0; b < MAXSIG; b++) {
      Bmat[a][b] = 0.0;
      Bset[a][b] = 0;
    }
}

PairSigmoidEpisig::~PairSigmoidEpisig()
{
  if (copymode) return;
  delete[] pvector;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(c0);
    memory->destroy(rc);
    memory->destroy(mu);
    memory->destroy(offset);
    memory->destroy(rcinv2);
    memory->destroy(nexp);
  }
}

/* ---------------------------------------------------------------------- */

void PairSigmoidEpisig::set_nsig(int n)
{
  if (n < 1 || n > MAXSIG)
    error->all(FLERR, "pair sigmoid/episig: Nsig must be between 1 and {}", (int) MAXSIG);
  if (nsig == n && pvector) return;
  nsig = n;
  nextra = nsig * (nsig + 1) / 2 + 1;    // Phi_ab (a<=b) + N_c
  delete[] pvector;
  pvector = new double[nextra];
  for (int k = 0; k < nextra; k++) pvector[k] = 0.0;
}

double **PairSigmoidEpisig::get_signals()
{
  int flag, cols;
  int idx = atom->find_custom("signals", flag, cols);
  if (idx < 0 || flag != 1)
    error->all(FLERR, "pair sigmoid/episig: per-atom array 'd2_signals' not found");
  if (cols != nsig)
    error->all(FLERR, "pair sigmoid/episig: d2_signals has {} columns, pair_style says {}", cols,
               nsig);
  return atom->darray[idx];
}

/* ---------------------------------------------------------------------- */

template <int FORM, int NS> void PairSigmoidEpisig::eval(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, itype, jtype;
  double xtmp, ytmp, ztmp, delx, dely, delz, evdwl, fpair, rsq, r, factor_lj;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  // re-fetch every call: the per-atom array may be reallocated when atoms migrate
  double **sig = get_signals();

  const int ntri = nsig * (nsig + 1) / 2;
  const int accumulate = eflag_global;    // compute pair only reads pvector on energy steps
  if (accumulate)
    for (int k = 0; k < nextra; k++) pvector[k] = 0.0;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // local copy of B: no aliasing with force arrays, fixed bounds when NS > 0 (unrolled loops)
  const int ns = NS ? NS : nsig;
  double Bl[MAXSIG][MAXSIG];
  for (int a = 0; a < ns; a++)
    for (int b = 0; b < ns; b++) Bl[a][b] = Bmat[a][b];

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    const double *si = sig[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;
      jtype = type[j];
      if (rsq >= cutsq[itype][jtype]) continue;

      // S_ij = x_i^T B x_j
      const double *sj = sig[j];
      double Sij = 0.0;
      for (int a = 0; a < ns; a++) {
        double s = 0.0;
        for (int b = 0; b < ns; b++) s += Bl[a][b] * sj[b];
        Sij += si[a] * s;
      }
      const double amp = c0[itype][jtype] + Sij;    // well depth for this pair

      double g;
      if (FORM == RATIONAL) {
        // f = 1/(1+u^n), u = r^2/rc^2 ;  -(dE/dr)/r = -amp * 2n u^(n-1) / (rc^2 (1+u^n)^2)
        const double u = rsq * rcinv2[itype][jtype];
        const int n = nexp[itype][jtype];
        double un1 = 1.0;
        for (int p = 1; p < n; p++) un1 *= u;
        const double den = 1.0 / (1.0 + un1 * u);
        g = den - offset[itype][jtype];
        fpair = -factor_lj * amp * 2.0 * n * un1 * rcinv2[itype][jtype] * den * den;
      } else {
        r = sqrt(rsq);
        double th;
        if (FORM == EXPF) {
          const double e = exp(-2.0 * mu[itype][jtype] * (rc[itype][jtype] - r));
          th = (1.0 - e) / (1.0 + e);
        } else {
          th = tanh(mu[itype][jtype] * (rc[itype][jtype] - r));
        }
        g = 0.5 * (1.0 + th) - offset[itype][jtype];
        // E = -amp*g ;  dE/dr = amp * (mu/2) * (1 - th^2) ;  fpair = -(dE/dr)/r
        fpair = -factor_lj * amp * 0.5 * mu[itype][jtype] * (1.0 - th * th) / r;
      }

      f[i][0] += delx * fpair;
      f[i][1] += dely * fpair;
      f[i][2] += delz * fpair;
      if (newton_pair || j < nlocal) {
        f[j][0] -= delx * fpair;
        f[j][1] -= dely * fpair;
        f[j][2] -= delz * fpair;
      }

      if (eflag) evdwl = -factor_lj * amp * g;

      if (accumulate) {
        // pair owned once with newton on; split between ranks with newton off
        const double w = factor_lj * g * ((newton_pair || j < nlocal) ? 1.0 : 0.5);
        for (int a = 0; a < ns; a++) {
          pvector[tri_index(a, a)] += w * si[a] * sj[a];
          for (int b = a + 1; b < ns; b++)
            pvector[tri_index(a, b)] += w * (si[a] * sj[b] + si[b] * sj[a]);
        }
        pvector[ntri] += w;
      }

      if (evflag) ev_tally(i, j, nlocal, newton_pair, evdwl, 0.0, fpair, delx, dely, delz);
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ---------------------------------------------------------------------- */

void PairSigmoidEpisig::compute(int eflag, int vflag)
{
  if (nsig == 3) {
    if (form == RATIONAL) eval<RATIONAL, 3>(eflag, vflag);
    else if (form == EXPF) eval<EXPF, 3>(eflag, vflag);
    else eval<TANH, 3>(eflag, vflag);
  } else {
    if (form == RATIONAL) eval<RATIONAL, 0>(eflag, vflag);
    else if (form == EXPF) eval<EXPF, 0>(eflag, vflag);
    else eval<TANH, 0>(eflag, vflag);
  }
}

/* ---------------------------------------------------------------------- */

double PairSigmoidEpisig::switch_f(int i, int j, double r) const
{
  if (form == RATIONAL) {
    double u = r * r * rcinv2[i][j], un = 1.0;
    for (int p = 0; p < nexp[i][j]; p++) un *= u;
    return 1.0 / (1.0 + un);
  }
  return 0.5 * (1.0 + tanh(mu[i][j] * (rc[i][j] - r)));
}

/* ---------------------------------------------------------------------- */

void PairSigmoidEpisig::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;
  memory->create(setflag, n, n, "pair:setflag");
  for (int i = 1; i < n; i++)
    for (int j = i; j < n; j++) setflag[i][j] = 0;
  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(cut, n, n, "pair:cut");
  memory->create(c0, n, n, "pair:c0");
  memory->create(rc, n, n, "pair:rc");
  memory->create(mu, n, n, "pair:mu");
  memory->create(offset, n, n, "pair:offset");
  memory->create(rcinv2, n, n, "pair:rcinv2");
  memory->create(nexp, n, n, "pair:nexp");
}

/* pair_style sigmoid/episig Nsig rcut */

void PairSigmoidEpisig::settings(int narg, char **arg)
{
  if (narg != 2 && narg != 4)
    error->all(FLERR, "Illegal pair_style sigmoid/episig command: Nsig rcut [form tanh|exp|rational]");
  set_nsig(utils::inumeric(FLERR, arg[0], false, lmp));
  cut_global = utils::numeric(FLERR, arg[1], false, lmp);
  form = TANH;
  if (narg == 4) {
    if (strcmp(arg[2], "form") != 0) error->all(FLERR, "pair_style sigmoid/episig: unknown keyword {}", arg[2]);
    if (strcmp(arg[3], "tanh") == 0) form = TANH;
    else if (strcmp(arg[3], "exp") == 0) form = EXPF;
    else if (strcmp(arg[3], "rational") == 0) form = RATIONAL;
    else error->all(FLERR, "pair_style sigmoid/episig: unknown form {}", arg[3]);
  }

  if (allocated) {
    for (int i = 1; i <= atom->ntypes; i++)
      for (int j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ---------------------------------------------------------------------- */

void PairSigmoidEpisig::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  // B-matrix entry:  "* * B a b value"  (also legacy "B a b value")
  int boff = -1;
  if (narg >= 1 && strcmp(arg[0], "B") == 0) boff = 0;
  else if (narg >= 3 && strcmp(arg[2], "B") == 0) boff = 2;

  if (boff >= 0) {
    if (narg != boff + 4) error->all(FLERR, "pair_coeff B: expected 'B a b value'");
    int a = utils::inumeric(FLERR, arg[boff + 1], false, lmp) - 1;
    int b = utils::inumeric(FLERR, arg[boff + 2], false, lmp) - 1;
    if (a < 0 || a >= nsig || b < 0 || b >= nsig)
      error->all(FLERR, "pair_coeff B: signal index out of range 1..{}", nsig);
    double v = utils::numeric(FLERR, arg[boff + 3], false, lmp);
    Bmat[a][b] = Bmat[b][a] = v;
    Bset[a][b] = Bset[b][a] = 1;
    return;
  }

  // type-pair coefficients:  I J C rc mu [rcut]
  if (narg < 5 || narg > 6) error->all(FLERR, "pair_coeff sigmoid/episig: I J C rc mu [rcut]");
  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
  double c_one = utils::numeric(FLERR, arg[2], false, lmp);
  double rc_one = utils::numeric(FLERR, arg[3], false, lmp);
  double mu_one = utils::numeric(FLERR, arg[4], false, lmp);
  double cut_one = (narg == 6) ? utils::numeric(FLERR, arg[5], false, lmp) : cut_global;
  if (mu_one <= 0.0) error->all(FLERR, "pair_coeff sigmoid/episig: mu must be > 0");

  int count = 0;
  for (int i = ilo; i <= ihi; i++)
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      c0[i][j] = c_one;
      rc[i][j] = rc_one;
      mu[i][j] = mu_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ---------------------------------------------------------------------- */

void PairSigmoidEpisig::init_style()
{
  if (nsig < 1) error->all(FLERR, "pair sigmoid/episig: Nsig not set");
  for (int a = 0; a < nsig; a++)
    for (int b = a; b < nsig; b++)
      if (!Bset[a][b]) error->all(FLERR, "pair sigmoid/episig: B[{}][{}] not specified", a + 1, b + 1);
  get_signals();    // fail early if the property is missing / wrong width
  neighbor->add_request(this);
}

double PairSigmoidEpisig::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {    // simple arithmetic mixing
    c0[i][j] = 0.5 * (c0[i][i] + c0[j][j]);
    rc[i][j] = 0.5 * (rc[i][i] + rc[j][j]);
    mu[i][j] = 0.5 * (mu[i][i] + mu[j][j]);
    cut[i][j] = mix_distance(cut[i][i], cut[j][j]);
  }
  rcinv2[i][j] = 1.0 / (rc[i][j] * rc[i][j]);
  nexp[i][j] = static_cast<int>(lround(mu[i][j]));
  if (form == RATIONAL && (nexp[i][j] < 1 || fabs(mu[i][j] - nexp[i][j]) > 1e-12))
    error->all(FLERR, "pair sigmoid/episig form rational: mu slot must be a positive integer n");
  if (offset_flag && cut[i][j] > 0.0)
    offset[i][j] = switch_f(i, j, cut[i][j]);
  else
    offset[i][j] = 0.0;

  c0[j][i] = c0[i][j];
  rc[j][i] = rc[i][j];
  mu[j][i] = mu[i][j];
  cut[j][i] = cut[i][j];
  offset[j][i] = offset[i][j];
  rcinv2[j][i] = rcinv2[i][j];
  nexp[j][i] = nexp[i][j];
  return cut[i][j];
}

/* ---------------------------------------------------------------------- */

void PairSigmoidEpisig::write_restart(FILE *fp)
{
  write_restart_settings(fp);
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j], sizeof(int), 1, fp);
      if (setflag[i][j]) {
        fwrite(&c0[i][j], sizeof(double), 1, fp);
        fwrite(&rc[i][j], sizeof(double), 1, fp);
        fwrite(&mu[i][j], sizeof(double), 1, fp);
        fwrite(&cut[i][j], sizeof(double), 1, fp);
      }
    }
}

void PairSigmoidEpisig::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();
  int me = comm->me;
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR, &setflag[i][j], sizeof(int), 1, fp, nullptr, error);
      MPI_Bcast(&setflag[i][j], 1, MPI_INT, 0, world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR, &c0[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &rc[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &mu[i][j], sizeof(double), 1, fp, nullptr, error);
          utils::sfread(FLERR, &cut[i][j], sizeof(double), 1, fp, nullptr, error);
        }
        MPI_Bcast(&c0[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&rc[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&mu[i][j], 1, MPI_DOUBLE, 0, world);
        MPI_Bcast(&cut[i][j], 1, MPI_DOUBLE, 0, world);
      }
    }
}

void PairSigmoidEpisig::write_restart_settings(FILE *fp)
{
  fwrite(&nsig, sizeof(int), 1, fp);
  fwrite(&cut_global, sizeof(double), 1, fp);
  fwrite(&offset_flag, sizeof(int), 1, fp);
  fwrite(&mix_flag, sizeof(int), 1, fp);
  fwrite(&Bmat[0][0], sizeof(double), MAXSIG * MAXSIG, fp);
  fwrite(&Bset[0][0], sizeof(int), MAXSIG * MAXSIG, fp);
  fwrite(&form, sizeof(int), 1, fp);
}

void PairSigmoidEpisig::read_restart_settings(FILE *fp)
{
  int me = comm->me, n = 0;
  if (me == 0) {
    utils::sfread(FLERR, &n, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &cut_global, sizeof(double), 1, fp, nullptr, error);
    utils::sfread(FLERR, &offset_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &mix_flag, sizeof(int), 1, fp, nullptr, error);
    utils::sfread(FLERR, &Bmat[0][0], sizeof(double), MAXSIG * MAXSIG, fp, nullptr, error);
    utils::sfread(FLERR, &Bset[0][0], sizeof(int), MAXSIG * MAXSIG, fp, nullptr, error);
    utils::sfread(FLERR, &form, sizeof(int), 1, fp, nullptr, error);
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, world);
  MPI_Bcast(&cut_global, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&offset_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&mix_flag, 1, MPI_INT, 0, world);
  MPI_Bcast(&Bmat[0][0], MAXSIG * MAXSIG, MPI_DOUBLE, 0, world);
  MPI_Bcast(&Bset[0][0], MAXSIG * MAXSIG, MPI_INT, 0, world);
  MPI_Bcast(&form, 1, MPI_INT, 0, world);
  set_nsig(n);
}

void PairSigmoidEpisig::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp, "%d %g %g %g\n", i, c0[i][i], rc[i][i], mu[i][i]);
}

void PairSigmoidEpisig::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp, "%d %d %g %g %g %g\n", i, j, c0[i][j], rc[i][j], mu[i][j], cut[i][j]);
}

/* i, j are local indices (used by compute pair/local, pair_write needs real atoms) */

double PairSigmoidEpisig::single(int i, int j, int itype, int jtype, double rsq,
                                 double /*factor_coul*/, double factor_lj, double &fforce)
{
  double **sig = get_signals();
  double Sij = 0.0;
  for (int a = 0; a < nsig; a++)
    for (int b = 0; b < nsig; b++) Sij += sig[i][a] * Bmat[a][b] * sig[j][b];
  const double amp = c0[itype][jtype] + Sij;
  if (form == RATIONAL) {
    const double u = rsq * rcinv2[itype][jtype];
    const int n = nexp[itype][jtype];
    double un1 = 1.0;
    for (int p = 1; p < n; p++) un1 *= u;
    const double den = 1.0 / (1.0 + un1 * u);
    fforce = -factor_lj * amp * 2.0 * n * un1 * rcinv2[itype][jtype] * den * den;
    return -factor_lj * amp * (den - offset[itype][jtype]);
  }
  const double r = sqrt(rsq);
  const double th = tanh(mu[itype][jtype] * (rc[itype][jtype] - r));
  fforce = -factor_lj * amp * 0.5 * mu[itype][jtype] * (1.0 - th * th) / r;
  return -factor_lj * amp * (0.5 * (1.0 + th) - offset[itype][jtype]);
}

void *PairSigmoidEpisig::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str, "c0") == 0) return (void *) c0;
  if (strcmp(str, "rc") == 0) return (void *) rc;
  if (strcmp(str, "mu") == 0) return (void *) mu;
  return nullptr;
}
