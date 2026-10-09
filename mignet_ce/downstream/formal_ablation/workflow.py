"""One-command formal ablation pipeline: fixed protocol -> cache -> CSV/PDF/PNG.

No model selection and no dependency on the old mignet_ce.pij.ablation registry.
On missing data the scientific run fails instead of silently filling values.
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import pandas as pd

from mignet_ce.io.loaders import LayerDataResolver
from .config import FormalAblationConfig, HIERARCHIES, TIME_POINTS
from .evaluator import build_cost_blocks, evaluate_cost_blocks, load_context
from .tables import build_tables
from .plots import render_table

PROTOCOL = 'formal_table123_fixed_v1'
EXPECTED_ROWS_PER_HIERARCHY = 9 * 3  # T1 4 + T2 2 + T3 3, times three time pairs

def _json(data):
    return json.dumps(data, ensure_ascii=False,sort_keys=True,default=str,separators=(',',':'))

def _sha(data):
    return hashlib.sha256(_json(data).encode('utf-8')).hexdigest()

def _code_fingerprint():
    # Scientific code files. Hashes change when exact feature/PIJ implementation changes.
    root=Path(__file__).resolve().parents[3]
    files=[
        'mignet_ce/downstream/formal_ablation/config.py',
        'mignet_ce/downstream/formal_ablation/evaluator.py',
        'mignet_ce/pij/compare/_shared/features.py',
        'mignet_ce/pij/compare/_shared/kl.py',
        'mignet_ce/pij/compare/_shared/cosine.py',
        'mignet_ce/pij/compare/_shared/ng_kl_ot.py',
        'mignet_ce/pij/compare/_shared/log_balanced_ot.py',
        'mignet_ce/grn_representation/config.py',
        'mignet_ce/grn_representation/basis.py',
        'mignet_ce/grn_representation/encoder.py',
        'mignet_ce/networks/light_cci_grn.py',
        'mignet_ce/networks/light_cci.py',
        'mignet_ce/metrics.py',
        'mignet_ce/config.py',
    ]
    out={}
    for f in files:
        path=root/f
        if not path.is_file():raise FileNotFoundError(f'Required scientific source file absent: {path}')
        out[f]=hashlib.sha256(path.read_bytes()).hexdigest()
    return out

def _input_fingerprint(cfg, hierarchy):
    resolver=LayerDataResolver(cfg.data_root)
    paths=[]
    for stage in TIME_POINTS:
        for layer in hierarchy[1:]:
            p=resolver.paths(layer,cfg.organ,stage)
            for path in (p.h5ad,p.cci_total,p.cci_index,p.grn_edges):
                if not path.is_file():
                    raise FileNotFoundError(f'Input missing for {hierarchy[0]}, E{stage}: {path}')
                stat=path.stat()
                paths.append({'path':str(path.resolve()),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
    return paths

def _write_json(path, payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    tmp.replace(path)

def _cache_name(hierarchy):
    return hierarchy[0].lower().replace(' ','').replace('->','_to_')

def run(cfg:FormalAblationConfig, *, mode:str='full') -> dict:
    cfg.validate()
    if mode not in {'full','resume','render'}:raise ValueError(f'Unknown mode: {mode}')
    output=cfg.output_root
    selected_hierarchies=cfg.selected_hierarchies()
    manifest_path=output/'manifest.json'
    rows=[]; reused=0;computed=0
    if mode=='render':
        if not manifest_path.exists():raise FileNotFoundError('Cannot render without validated full-run manifest.')
        old=json.loads(manifest_path.read_text(encoding='utf-8'))
        if old['contract'] != cfg.public_contract() or old['protocol'] != PROTOCOL:
            raise RuntimeError('Render-only manifest protocol/config mismatch.')
        raw_path=output/'results'/'formal_ablation_long.csv'
        if not raw_path.is_file():raise FileNotFoundError(raw_path)
        data=pd.read_csv(raw_path)
    else:
        code_sig=_code_fingerprint()
        for hierarchy in selected_hierarchies:
            key=_cache_name(hierarchy)
            cache=output/'cache'/(key+'.json')
            inputs=_input_fingerprint(cfg,hierarchy)
            fingerprint=_sha({'protocol':PROTOCOL,'contract':cfg.public_contract(),
                              'code_sig':code_sig,'input_sig':inputs,'hierarchy':hierarchy})
            if cache.is_file():
                saved=json.loads(cache.read_text(encoding='utf-8'))
                if (saved.get('fingerprint')==fingerprint and
                        len(saved.get('rows',[]))==EXPECTED_ROWS_PER_HIERARCHY):
                    rows.extend(saved['rows']);reused+=1
                    print(f'[formal_ablation] reuse {hierarchy[0]}',flush=True)
                    continue
            print(f'[formal_ablation] compute {hierarchy[0]}',flush=True)
            context=load_context(cfg,hierarchy)
            costs=build_cost_blocks(cfg,hierarchy,context)
            new_rows=evaluate_cost_blocks(cfg,hierarchy,costs)
            if len(new_rows)!=EXPECTED_ROWS_PER_HIERARCHY:
                raise RuntimeError(f'Incomplete computed rows: {hierarchy[0]} {len(new_rows)}')
            _write_json(cache,{'fingerprint':fingerprint,'hierarchy':hierarchy[0],
                               'rows':new_rows,'inputs':inputs,'code_fingerprint':_sha(code_sig)})
            rows.extend(new_rows);computed+=1
        data=pd.DataFrame(rows)
        if len(data)!=len(selected_hierarchies)*EXPECTED_ROWS_PER_HIERARCHY:
            raise RuntimeError(f'Incomplete full run: {len(data)} rows; expected {len(selected_hierarchies)*EXPECTED_ROWS_PER_HIERARCHY}.')
        target=output/'results'/'formal_ablation_long.csv'
        target.parent.mkdir(parents=True,exist_ok=True)
        data.to_csv(target,index=False,float_format='%.12g')
        # The manifest is written after all tables have passed validation below.
    tables=build_tables(data,hierarchies=selected_hierarchies)
    for name,frame in tables.items():
        table_dir=output/'tables';table_dir.mkdir(parents=True,exist_ok=True)
        frame.to_csv(table_dir/f'{name}.csv',index=False,float_format='%.12g')
        render_table(frame,name,table_dir/f'{name}.png',dpi=cfg.dpi)
    result={'protocol':PROTOCOL,'contract':cfg.public_contract(),
            'rows':len(data),'cache_reused':reused,'hierarchies_computed':computed,
            'status':'complete','input_signature_policy':'size+mtime_ns+absolute_path; code_sha256',
            'report':'independent_hierarchy_PIJ_direct_delta_EI_not_induced_macro_EI',
            'tables':[f'tables/{t}.csv' for t in ('table1','table2','table3')],
            'display_mean_column':False, 'selection_performed':False}
    if mode!='render':_write_json(manifest_path,result)
    print(f'[formal_ablation] completed: {len(data)} condition/time/hierarchy rows; '
          f'cache reused {reused}, computed {computed}; tables in {output / "tables"}',flush=True)
    return result
