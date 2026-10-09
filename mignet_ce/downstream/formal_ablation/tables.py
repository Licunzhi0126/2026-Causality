"""Deterministic three-table renderer inputs. No selection, no mean column."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import HIERARCHIES, TIME_PAIRS, T1_CONFIGS, FIXED_GRN, FIXED_CCI, FIXED_PIJ

VALUE_COLS = TIME_PAIRS

def build_tables(raw: pd.DataFrame, *, allow_incomplete: bool = False, hierarchies=None) -> dict[str,pd.DataFrame]:
    need = {'table','hierarchy','time_pair','grn_feature','cci_feature','pij','input','delta_EI'}
    if not need.issubset(raw.columns): raise ValueError(f'Missing raw columns: {sorted(need - set(raw.columns))}')
    data = raw.copy()
    identity = ['table','hierarchy','time_pair','grn_feature','cci_feature','pij','input']
    if data.duplicated(identity).any(): raise ValueError('Duplicate metric conditions; refusing ambiguous table cell.')
    def one(t, hierarchy, pair, g,c,pij,inp):
        found=data.loc[data['table'].eq(t)&data['hierarchy'].eq(hierarchy)&data['time_pair'].eq(pair)&data['grn_feature'].eq(g)&data['cci_feature'].eq(c)&data['pij'].eq(pij)&data['input'].eq(inp)]
        if len(found)!=1:
            if allow_incomplete and len(found)==0:return float('nan')
            raise ValueError(f'Expected one {t}: {hierarchy},{pair},{g},{c},{pij},{inp}; got {len(found)}')
        value=float(found.iloc[0]['delta_EI'])
        if not np.isfinite(value):raise ValueError(f'Nonfinite delta_EI: {t},{hierarchy},{pair}')
        return value
    active_hierarchies = HIERARCHIES if hierarchies is None else tuple(hierarchies)
    out={}
    for table in ('table1','table2','table3'):
        rows=[]
        for hierarchy,_,_ in active_hierarchies:
            if table=='table1':
                specs = [(g,c,'KL+OT','GRN+CCI') for g,c in T1_CONFIGS]
            elif table=='table2':
                specs = [(FIXED_GRN,FIXED_CCI,p,'GRN+CCI') for p in ('KL','KL+OT')]
            else:
                specs = [(FIXED_GRN,FIXED_CCI,FIXED_PIJ,i) for i in ('GRN-only','CCI-only','GRN+CCI')]
            for g,c,p,i in specs:
                row={'Hierarchy':hierarchy}
                if table=='table1':row.update({'GRN Feature':g,'CCI Feature':c})
                elif table=='table2':row.update({'PIJ':p})
                else:row.update({'Input':i})
                row.update({pair:one(table,hierarchy,pair,g,c,p,i) for pair in TIME_PAIRS})
                rows.append(row)
        out[table]=pd.DataFrame(rows)
    return out
