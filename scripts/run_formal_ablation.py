#!/usr/bin/env python3
"""One command runs all three fixed-paper ablations. No dynamic configuration selection.

python scripts/run_formal_ablation.py --data-root /path/to/E1S1_domain_factory \\
  --output-root /path/to/outputs/formal_ablation --mode full
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
REPO_ROOT=Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:sys.path.insert(0,str(REPO_ROOT))
from mignet_ce.config import DEFAULT_DATA_ROOT
from mignet_ce.downstream.formal_ablation import FormalAblationConfig
from mignet_ce.downstream.formal_ablation.workflow import run

def main(argv=None):
    p=argparse.ArgumentParser(description='Fixed GRN/CCI Feature × PIJ × Input tables; no selection.')
    p.add_argument('--data-root',type=Path,default=DEFAULT_DATA_ROOT)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--organ',default='heart')
    p.add_argument('--mode',choices=['full','resume','render'],default='full')
    p.add_argument('--dpi',type=int,default=250)
    p.add_argument('--include-k10',action='store_true',help='Include K10 hierarchies when corresponding real data are present.')
    a=p.parse_args(argv)
    cfg=FormalAblationConfig(data_root=a.data_root,output_root=a.output_root,
                             organ=a.organ,dpi=a.dpi,include_k10=a.include_k10)
    run(cfg,mode=a.mode)
    return 0
if __name__=='__main__':raise SystemExit(main())
