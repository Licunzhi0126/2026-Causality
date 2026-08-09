#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mignet_ce.visualization.downstream.unified_suite import UnifiedSuiteConfig, mapping_record, candidate_sweep, MAPPINGS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-root',type=Path,required=True); ap.add_argument('--closure-cache-root',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--k-grid',nargs='+',type=int,required=True); ap.add_argument('--seeds',nargs='+',type=int,required=True); ap.add_argument('--epochs',type=int,required=True); ap.add_argument('--crossfit-folds',type=int,default=5)
    a=ap.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    cfg=UnifiedSuiteConfig(a.data_root,a.closure_cache_root,a.output_dir,candidate_k_grid=tuple(a.k_grid),candidate_seeds=tuple(a.seeds),candidate_epochs=a.epochs,crossfit_folds=a.crossfit_folds,run_candidate_sweep=True)
    rec={(m,p):mapping_record(cfg,m,p) for p in cfg.adjacent_pairs for m in MAPPINGS}
    df=candidate_sweep(cfg,rec); df.to_csv(a.output_dir/'pareto_screening.csv',index=False); print(a.output_dir/'pareto_screening.csv')
if __name__=='__main__': main()
