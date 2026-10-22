#!/usr/bin/env python3
"""Compare Langevin and NVT runs from the thermostat sweep CSV."""
import csv, math
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path('results/thermostat_sweep_dt0p005_fixed')
SOURCE = ROOT / 'sweep_results.csv'
OUT = ROOT / 'plots'
TARGET = 1.0
FIELDS = ('Step', 'simulation_time', 'Temp', 'PotEng', 'TotEng', 'E_vdwl')

def load_runs():
    runs = defaultdict(list)
    with SOURCE.open(newline='') as f:
        for row in csv.DictReader(f):
            try:
                for k in FIELDS:
                    row[k] = float(row[k]) if row.get(k, '').strip() else math.nan
                row['B_common'] = float(row['B_common'])
            except (ValueError, TypeError):
                continue
            # Skip minimization/incomplete temperature records.
            if not math.isfinite(row['Temp']) or row['Temp'] <= 0:
                continue
            row['thermostat'] = row['thermostat'].lower()
            runs[(row['B_common'], row['thermostat'])].append(row)
    for rr in runs.values():
        rr.sort(key=lambda r: (r['simulation_time'], r['Step']))
    return runs

def main():
    if not SOURCE.exists():
        raise SystemExit(f'Cannot find {SOURCE}; run from ~/19Oct_sweep')
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    Bs = sorted({b for b, _ in runs})
    styles = {'langevin': '-', 'nvt': '--'}
    def save(fig, name):
        fig.tight_layout(); fig.savefig(OUT/name, dpi=180, bbox_inches='tight'); plt.close(fig)
        print('Wrote', OUT/name)

    # 1. Temperature time series, one panel per B.
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, b in zip(axs.flat, Bs):
        for th in ('langevin', 'nvt'):
            rr = runs.get((b, th), [])
            if rr:
                ax.plot([r['simulation_time'] for r in rr], [r['Temp'] for r in rr],
                        marker='o', ms=3, ls=styles[th], label=th.upper())
        ax.axhline(TARGET, color='black', lw=1, alpha=.7)
        ax.set(title=f'B = {b:g}', xlabel='Simulation time (LJ units)'); ax.grid(alpha=.25)
    for ax in axs[:, 0]: ax.set_ylabel('Temperature')
    axs.flat[0].legend(frameon=False); fig.suptitle('Temperature vs time: Langevin and NVT', y=1.02)
    save(fig, '01_temperature_timeseries.png')

    # 2. Temperature deviation from target.
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, b in zip(axs.flat, Bs):
        for th in ('langevin', 'nvt'):
            rr = runs.get((b, th), [])
            if rr:
                ax.plot([r['simulation_time'] for r in rr], [r['Temp']-TARGET for r in rr],
                        marker='o', ms=3, ls=styles[th], label=th.upper())
        ax.axhline(0, color='black', lw=1, alpha=.7)
        ax.set(title=f'B = {b:g}', xlabel='Simulation time (LJ units)'); ax.grid(alpha=.25)
    for ax in axs[:, 0]: ax.set_ylabel(r'$T - T_{target}$')
    axs.flat[0].legend(frameon=False); fig.suptitle('Temperature deviation from target T = 1', y=1.02)
    save(fig, '02_temperature_deviation.png')

    # 3. Mean T and observed min-max range, plus MAE summary.
    summary = []
    fig, ax = plt.subplots(figsize=(9, 5.5)); xs = list(range(len(Bs))); width = .34
    for offset, th in [(-width/2, 'langevin'), (width/2, 'nvt')]:
        means, lows, highs = [], [], []
        for b in Bs:
            rr = runs.get((b, th), [])
            if not rr:
                means.append(math.nan); lows.append(0); highs.append(0); continue
            ts = [r['Temp'] for r in rr]; mean = sum(ts)/len(ts)
            mae = sum(abs(t-TARGET) for t in ts)/len(ts)
            means.append(mean); lows.append(mean-min(ts)); highs.append(max(ts)-mean)
            summary.append([b, th, len(ts), mean, min(ts), max(ts), mae])
        ax.bar([x+offset for x in xs], means, width=width, label=th.upper())
        ax.errorbar([x+offset for x in xs], means, yerr=[lows, highs], fmt='none', ecolor='black', capsize=3)
    ax.axhline(TARGET, color='black', ls=':', lw=1, label='Target T = 1')
    ax.set_xticks(xs, [f'{b:g}' for b in Bs]); ax.set(xlabel='Common B', ylabel='Temperature', title='Mean temperature and sampled min-max range')
    ax.grid(axis='y', alpha=.25); ax.legend(frameon=False, ncols=3); save(fig, '03_temperature_summary.png')
    with (OUT/'temperature_summary.csv').open('w', newline='') as f:
        w=csv.writer(f); w.writerow(['B_common','thermostat','n_samples','mean_T','min_T','max_T','mean_abs_T_error'])
        for row in summary: w.writerow([row[0],row[1],row[2]]+[f'{v:.8g}' for v in row[3:]])

    # 4 and 5. Potential and van der Waals energy time series.
    for field, filename, title, ylabel in [
        ('PotEng','04_potential_energy.png','Potential energy','PotEng'),
        ('E_vdwl','05_vdwl_energy.png','Van der Waals energy',r'$E_{vdwl}$')]:
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        for ax,b in zip(axs.flat,Bs):
            for th in ('langevin','nvt'):
                rr=[r for r in runs.get((b,th),[]) if math.isfinite(r[field])]
                if rr: ax.plot([r['simulation_time'] for r in rr],[r[field] for r in rr],marker='o',ms=3,ls=styles[th],label=th.upper())
            ax.set(title=f'B = {b:g}',xlabel='Simulation time (LJ units)'); ax.grid(alpha=.25)
        for ax in axs[:,0]: ax.set_ylabel(ylabel)
        axs.flat[0].legend(frameon=False); fig.suptitle(f'{title}: Langevin and NVT',y=1.02); save(fig,filename)

    # 6. NVT - Langevin temperature difference at matched B; linear interpolation only within data range.
    fig, ax = plt.subplots(figsize=(9,5.5)); colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i,b in enumerate(Bs):
        lrr,nrr=runs.get((b,'langevin'),[]),runs.get((b,'nvt'),[])
        if not lrr or not nrr: continue
        nt=[r['simulation_time'] for r in nrr]; ntemp=[r['Temp'] for r in nrr]; tx=[]; delta=[]
        for r in lrr:
            t=r['simulation_time']
            if t<nt[0] or t>nt[-1]: continue
            j=next((j for j in range(1,len(nt)) if nt[j]>=t),None)
            if j is None: val=ntemp[-1]
            elif nt[j]==nt[j-1]: val=ntemp[j]
            else:
                frac=(t-nt[j-1])/(nt[j]-nt[j-1]); val=ntemp[j-1]+frac*(ntemp[j]-ntemp[j-1])
            tx.append(t); delta.append(val-r['Temp'])
        if tx: ax.plot(tx,delta,marker='o',ms=3,label=f'B = {b:g}',color=colors[i%len(colors)])
    ax.axhline(0,color='black',lw=1); ax.set(xlabel='Simulation time (LJ units)',ylabel=r'$T_{NVT}-T_{Langevin}$',title='Temperature difference at matched B')
    ax.grid(alpha=.25); ax.legend(frameon=False,ncols=2); save(fig,'06_temperature_difference.png')
    print('Summary CSV:', OUT/'temperature_summary.csv')
    print('Done. Min-max bars show sampled ranges, not confidence intervals.')

if __name__=='__main__': main()
