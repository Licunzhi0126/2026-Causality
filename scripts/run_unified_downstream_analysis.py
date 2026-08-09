#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, hashlib, sys, subprocess, os
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.visualization.downstream.unified_suite import UnifiedSuiteConfig, build_all_tables, save_tables, MAPPINGS, mapping_record, candidate_sweep
from mignet_ce.visualization.downstream.unified_plots import render_all_unified_figures


def sha256(path: Path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def audit(tables, figs):
    checks=[]
    def ck(name,ok,detail=''): checks.append({'check':name,'passed':bool(ok),'detail':str(detail)})
    # every main comparison includes all three mappings
    for name in ['metrics','states','spatial_spots','closure','spatial','effective','mechanism','null','fate','perturbation','leakage_states','leakage_spots','repairability','weighting','bootstrap']:
        df=tables[name]; present=set(df['mapping'].dropna().astype(str)) if 'mapping' in df else set(); ck(f'{name}: all three mappings',set(MAPPINGS).issubset(present),sorted(present))
    c=tables['closure']; ck('information identity',bool((abs(c.I_available_bits-c.I_retained_bits-c.closure_leakage_bits)<1e-8).all()),float((abs(c.I_available_bits-c.I_retained_bits-c.closure_leakage_bits)).max()))
    low=c.signal_status.eq('low-signal'); ck('low-signal quality is NA',bool(c.loc[low,'closure_quality'].isna().all()),int(low.sum()))
    r=tables['repairability']; step0=r[r.step==0].set_index(['mapping','time_pair']).leakage_bits; core=c.set_index(['mapping','time_pair']).closure_leakage_bits; common=step0.index.intersection(core.index); err=float((step0.loc[common]-core.loc[common]).abs().max()) if len(common) else float('nan'); ck('repairability step0 matches primary leakage',err<1e-8,err)
    p=tables['pareto']; ck('Pareto active K recorded',p.active_k.notna().all(),f'{p.active_k.min()}..{p.active_k.max()}'); ck('Optimized candidate cloud exists',(p.mapping=='Optimized candidate').sum()>=10,int((p.mapping=='Optimized candidate').sum()))
    sh=tables['soft_hard']; ck('soft assignments restricted to sensitivity table',len(sh)>0,len(sh))
    ck('exactly 15 unified PNG figures',len(figs)==15,len(figs)); ck('exactly 5 closure PNG figures',sum(Path(x).name.startswith('closure_') for x in figs)==5,sum(Path(x).name.startswith('closure_') for x in figs))
    return pd.DataFrame(checks)


def main():
    ap=argparse.ArgumentParser(description='Unified downstream analysis: K150, K40, and optimized coarse-graining in every main analysis.')
    ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--closure-cache-root',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--pareto-k-grid',nargs='+',type=int,default=[10,20,30,40,60,80,100,150]); ap.add_argument('--pareto-seeds',nargs='+',type=int,default=[42,43,44]); ap.add_argument('--candidate-epochs',type=int,default=80)
    ap.add_argument('--crossfit-folds',type=int,default=5); ap.add_argument('--null-repeats',type=int,default=200); ap.add_argument('--bootstrap-repeats',type=int,default=200); ap.add_argument('--skip-candidate-sweep',action='store_true')
    args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    cfg=UnifiedSuiteConfig(data_root=args.data_root,closure_cache_root=args.closure_cache_root,output_root=args.output_dir,crossfit_folds=args.crossfit_folds,null_repeats=args.null_repeats,bootstrap_repeats=args.bootstrap_repeats,perturb_random_repeats=20,repair_random_repeats=10,candidate_k_grid=tuple(args.pareto_k_grid),candidate_seeds=tuple(args.pareto_seeds),candidate_epochs=args.candidate_epochs,run_candidate_sweep=not args.skip_candidate_sweep)
    # Memory-isolated two-phase execution: candidate training in a child process,
    # then the complete core suite; every figure is rendered in its own child process.
    pareto_table=None
    env=dict(os.environ); code_root=str(Path(__file__).resolve().parents[1]); env['PYTHONPATH']=code_root + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    if cfg.run_candidate_sweep:
        cmd=[sys.executable,str(Path(__file__).with_name('run_unified_pareto_screen.py')),'--data-root',str(args.data_root),'--closure-cache-root',str(args.closure_cache_root),'--output-dir',str(args.output_dir),'--k-grid',*map(str,args.pareto_k_grid),'--seeds',*map(str,args.pareto_seeds),'--epochs',str(args.candidate_epochs),'--crossfit-folds',str(args.crossfit_folds)]
        subprocess.run(cmd,check=True,env=env)
        pareto_table=pd.read_csv(args.output_dir/'pareto_screening.csv')
    core_cfg=UnifiedSuiteConfig(**{**cfg.__dict__, 'run_candidate_sweep':False})
    tables,records=build_all_tables(core_cfg)
    if pareto_table is not None: tables['pareto']=pareto_table
    save_tables(tables,args.output_dir)
    figure_names=['causal_emergence_decomposition','state_level_ei_spatial','single_step_dynamics_overview','multistep_dynamics_overview','matched_random_null','cross_representation_consistency','effective_states_spatial','grn_cci_mechanism','macro_fate_paths','virtual_perturbation','closure_information_budget_and_operational_q','closure_causal_emergence_pareto','closure_horizon_ck_consistency','closure_failure_localization_and_repairability','closure_robustness_fairness_and_null']
    for name in figure_names:
        subprocess.run([sys.executable,str(Path(__file__).with_name('render_unified_downstream_figure.py')),'--results-root',str(args.output_dir),'--figure',name],check=True,env=env)
    figs=sorted((args.output_dir/'figures').glob('*.png'))
    audit_dir=args.output_dir/'audit'; audit_dir.mkdir(exist_ok=True); checks=audit(tables,figs); checks.to_csv(audit_dir/'validation_checks.csv',index=False)
    manifest={'mappings':list(MAPPINGS),'time_points':list(cfg.times),'pareto_k_grid':list(cfg.candidate_k_grid),'pareto_seeds':list(cfg.candidate_seeds),'candidate_epochs':cfg.candidate_epochs,'null_repeats':cfg.null_repeats,'bootstrap_repeats':cfg.bootstrap_repeats,'figure_count':len(figs),'closure_figure_count':sum(Path(x).name.startswith('closure_') for x in figs),'validation_passed':bool(checks.passed.all())}
    (audit_dir/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(checks.to_string(index=False)); print(json.dumps(manifest,indent=2))
    if not checks.passed.all(): sys.exit(2)
if __name__=='__main__': main()
