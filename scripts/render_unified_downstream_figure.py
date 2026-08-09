#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mignet_ce.visualization.downstream.unified_plots as p
FUNCTIONS={
'causal_emergence_decomposition':p.plot_causal_emergence_decomposition,
'state_level_ei_spatial':p.plot_state_level_ei_spatial,
'single_step_dynamics_overview':p.plot_single_step_dynamics,
'multistep_dynamics_overview':p.plot_multistep_dynamics,
'matched_random_null':p.plot_random_null,
'cross_representation_consistency':p.plot_consistency,
'effective_states_spatial':p.plot_effective_spatial,
'grn_cci_mechanism':p.plot_mechanism,
'macro_fate_paths':p.plot_fate,
'virtual_perturbation':p.plot_perturbation,
'closure_information_budget_and_operational_q':p.plot_closure_information_budget_operational,
'closure_causal_emergence_pareto':p.plot_closure_pareto,
'closure_horizon_ck_consistency':p.plot_closure_horizon,
'closure_failure_localization_and_repairability':p.plot_closure_failure_repair,
'closure_robustness_fairness_and_null':p.plot_closure_robustness,
}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results-root',type=Path,required=True); ap.add_argument('--figure',choices=FUNCTIONS,required=True); a=ap.parse_args()
    t={x.stem:pd.read_csv(x) for x in (a.results_root/'tables').glob('*.csv')}; (a.results_root/'figures').mkdir(exist_ok=True); FUNCTIONS[a.figure](t,a.results_root/'figures')
if __name__=='__main__': main()
