from __future__ import annotations

"""Unified downstream suite.

All downstream comparisons use the same three representations:
Seurat K150, Seurat K40, and complete_combined_coarse + light_cci_grn + NG_KLot.
Dynamic closure follows a hard-state, state-balanced information-budget primary
semantics.  Soft assignments are only used in an explicit sensitivity panel.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import json, math

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

from .dynamic_closure.analysis import (
    compact_hard_assignment as _compact_hard_assignment,
    crossfit_macro_q_error as _crossfit_macro_q_error,
    effective_information as _effective_information,
    entropy_rows as _entropy_rows,
    information_closure_budget as _information_closure_budget,
    information_closure_budget_from_observed as _information_closure_budget_from_observed,
    induced_macro_q as _induced_macro_q,
    js_rows as _js_rows,
    kl_rows as _kl_rows,
    row_normalize as _row_normalize,
    state_balanced_micro_weights as _state_balanced_micro_weights,
    state_level_ei as _state_level_ei,
    weighted_information as _weighted_information,
)

EPS = 1e-12
MAPPINGS = ("Seurat K150", "Seurat K40", "Optimized coarse-graining")
LAYER_BY_MAPPING = {"Seurat K150":"seurat_k150", "Seurat K40":"seurat_k40"}
COLORS = {"Seurat K150":"#2F74B8", "Seurat K40":"#D45959", "Optimized coarse-graining":"#4D9A73"}
MARKERS = {"Seurat K150":"o", "Seurat K40":"s", "Optimized coarse-graining":"D"}

@dataclass(frozen=True)
class UnifiedSuiteConfig:
    data_root: Path
    closure_cache_root: Path
    output_root: Path
    organ: str = "heart"
    times: tuple[str,...] = ("11.5","12.5","13.5","14.5")
    random_seed: int = 20260809
    null_repeats: int = 100
    bootstrap_repeats: int = 100
    perturb_random_repeats: int = 80
    crossfit_folds: int = 5
    repair_max_splits: int = 5
    repair_random_repeats: int = 20
    candidate_k_grid: tuple[int,...] = (10,20,30,40,60,80)
    candidate_seeds: tuple[int,...] = (42,43,44)
    candidate_epochs: int = 20
    candidate_nmf_max_iter: int = 40
    candidate_large_nmf_max_iter: int = 20
    run_candidate_sweep: bool = True

    @property
    def adjacent_pairs(self):
        return tuple(f"{a}->{b}" for a,b in zip(self.times[:-1], self.times[1:]))

# ---------- basic math ----------
def row_normalize(x: np.ndarray) -> np.ndarray:
    return _row_normalize(x)

def entropy_rows(p: np.ndarray) -> np.ndarray:
    return _entropy_rows(p)

def entropy_prob(p: np.ndarray) -> float:
    p=np.asarray(p,float); p=p[p>EPS]; return float(-np.sum(p*np.log2(p)))

def effective_information(p: np.ndarray) -> float:
    return _effective_information(p)

def state_ei(p: np.ndarray) -> np.ndarray:
    return _state_level_ei(p)

def kl_rows(p,q):
    return _kl_rows(p,q)

def js_rows(p,q):
    return _js_rows(p,q)

def hard_compact(s: np.ndarray):
    return _compact_hard_assignment(s)

def state_balanced_weights(h: np.ndarray) -> np.ndarray:
    return _state_balanced_micro_weights(h)

def induced_q(observed: np.ndarray,h: np.ndarray,w: np.ndarray):
    return _induced_macro_q(observed,h,w)

def weighted_information(rows: np.ndarray,w: np.ndarray) -> float:
    return _weighted_information(rows,w)

def closure_budget(p: np.ndarray, hs: np.ndarray, ht: np.ndarray, low_signal_threshold: float|None=None):
    budget=_information_closure_budget(p,hs,ht,low_signal_threshold_bits=low_signal_threshold)
    return dict(observed=budget['observed'],weights=budget['weights'],q_induced=budget['q_induced'],predicted=budget['predicted_induced'],
                I_available_bits=budget['I_available_bits'],I_retained_bits=budget['I_macro_retained_bits'],
                closure_leakage_bits=budget['closure_leakage_bits'],closure_quality=budget['macro_sufficiency'],
                signal_status=budget['signal_status'],signal_threshold_bits=budget['signal_threshold_bits'],
                identity_error=budget['information_identity_error'],source_hard=budget['source_hard'],target_hard=budget['target_hard'])


def closure_budget_from_observed(observed: np.ndarray, hs: np.ndarray, low_signal_threshold: float|None=None):
    """Primary hard/state-balanced closure budget when future hard-macro profiles are precomputed."""
    budget=_information_closure_budget_from_observed(observed,hs,low_signal_threshold_bits=low_signal_threshold)
    return dict(observed=budget['observed'],weights=budget['weights'],q_induced=budget['q_induced'],predicted=budget['predicted_induced'],
                I_available_bits=budget['I_available_bits'],I_retained_bits=budget['I_macro_retained_bits'],
                closure_leakage_bits=budget['closure_leakage_bits'],closure_quality=budget['macro_sufficiency'],
                signal_status=budget['signal_status'],signal_threshold_bits=budget['signal_threshold_bits'],
                identity_error=budget['information_identity_error'],source_hard=budget['source_hard'])

def crossfit_error(observed: np.ndarray,hs: np.ndarray,w:np.ndarray,folds=5,seed=20260809):
    result=_crossfit_macro_q_error(observed,hs,weights=w,folds=folds,seed=seed)
    return dict(crossfit_kl_bits=result['crossfit_kl_bits'],crossfit_js=result['crossfit_js'],
                crossfit_coverage=result['crossfit_coverage'],singleton_weight=result['singleton_weight_fraction'])

# ---------- data loading ----------
def load_npz(path: Path) -> np.ndarray:
    try: return np.asarray(sp.load_npz(path).toarray(),float)
    except Exception:
        z=np.load(path)
        if isinstance(z,np.ndarray): return np.asarray(z,float)
        key='pij' if 'pij' in z.files else z.files[0]; out=np.asarray(z[key],float); z.close(); return out

def _obs_index(path: Path):
    with h5py.File(path,'r') as f:
        key=f['obs'].attrs['_index']; vals=f['obs'][key][:]
        return [v.decode() if isinstance(v,(bytes,bytearray)) else str(v) for v in vals]

def _coords(path:Path):
    with h5py.File(path,'r') as f:
        if 'obsm' in f and 'spatial' in f['obsm']:
            return np.asarray(f['obsm/spatial'],float)
        obs=f['obs']
        if 'x' in obs and 'y' in obs: return np.column_stack([np.asarray(obs['x'],float),np.asarray(obs['y'],float)])
    return np.zeros((len(_obs_index(path)),2),float)

def layer_stem(layer,organ,time): return {"spot":"spot","seurat_k150":"seurat150","seurat_k40":"seurat"}[layer]+f"_{organ}_{time}"
def h5ad_path(cfg,layer,time): return Path(cfg.data_root)/layer/cfg.organ/f"{layer_stem(layer,cfg.organ,time)}.h5ad"
def index_path(cfg,layer,time): return Path(cfg.data_root)/'cci'/layer/f"{layer_stem(layer,cfg.organ,time)}_index.tsv"
def cci_path(cfg,layer,time): return Path(cfg.data_root)/'cci'/layer/f"{layer_stem(layer,cfg.organ,time)}_CCI_total.npz"
def map_path(cfg,layer,time): return Path(cfg.data_root)/layer/cfg.organ/f"{layer_stem(layer,cfg.organ,time)}_spot_domain_map.csv"
def grn_path(cfg,layer,time): return Path(cfg.data_root)/'grn'/layer/layer_stem(layer,cfg.organ,time)/'grn_edges.csv'
def read_index(path): return pd.read_csv(path,sep='\t').iloc[:,0].astype(str).tolist()

def natural_assignment(cfg,layer,time,spot_order=None):
    spots=_obs_index(h5ad_path(cfg,'spot',time)) if spot_order is None else list(map(str,spot_order)); states=read_index(index_path(cfg,layer,time))
    frame=pd.read_csv(map_path(cfg,layer,time)); sid='spot_id' if 'spot_id' in frame else ('cell_name' if 'cell_name' in frame else frame.columns[0]); sc='domain_id' if 'domain_id' in frame else frame.columns[-1]
    lookup=frame.astype({sid:str,sc:str}).drop_duplicates(sid).set_index(sid)[sc]; sm={s:i for i,s in enumerate(states)}
    labels=np.array([sm[str(lookup.loc[s])] for s in spots],int); h=np.zeros((len(spots),len(states)),float); h[np.arange(len(spots)),labels]=1; return h,spots,states

def optimal_dir(cfg,pair):
    a,b=pair.split('->'); return Path(cfg.closure_cache_root)/'optimal_coarse'/f'{a}_to_{b}'

def optimized_pair(cfg,pair):
    root=optimal_dir(cfg,pair); a,b=pair.split('->')
    st=np.load(root/'S_t.npy'); sp_=np.load(root/'S_tp.npy'); hs,asrc=hard_compact(st); ht,atgt=hard_compact(sp_)
    at=pd.read_csv(root/'assignments_t.csv'); ap=pd.read_csv(root/'assignments_tp.csv'); sid='spot_id' if 'spot_id' in at else at.columns[0]; tid='spot_id' if 'spot_id' in ap else ap.columns[0]
    q=row_normalize(np.load(root/'PIJ_macro_train.npy'))[np.ix_(asrc,atgt)]; q=row_normalize(q)
    p=row_normalize(load_npz(root/'PIJ_micro.npz'))
    return dict(hs=hs,ht=ht,active_s=asrc,active_t=atgt,spots_s=at[sid].astype(str).tolist(),spots_t=ap[tid].astype(str).tolist(),q=q,p=p,
                soft_s=row_normalize(st),soft_t=row_normalize(sp_),summary=json.loads((root/'summary.json').read_text()),coords_s=np.load(root/'coords_t.npy'),coords_t=np.load(root/'coords_tp.npy'))

def natural_q(cfg,mapping,pair):
    layer=LAYER_BY_MAPPING[mapping]; a,b=pair.split('->')
    return row_normalize(np.load(Path(cfg.closure_cache_root)/'tables'/f'q_direct_{layer}_{a}_to_{b}.npy'))
def spot_p(cfg,pair):
    a,b=pair.split('->'); return row_normalize(load_npz(Path(cfg.closure_cache_root)/'matrices'/f'pij_spot_{a}_to_{b}.npz'))

def mapping_record(cfg,mapping,pair):
    a,b=pair.split('->')
    if mapping=="Optimized coarse-graining":
        o=optimized_pair(cfg,pair); return dict(mapping=mapping,pair=pair,p=o['p'],hs=o['hs'],ht=o['ht'],q_direct=o['q'],spots_s=o['spots_s'],spots_t=o['spots_t'],soft_s=o['soft_s'],soft_t=o['soft_t'],summary=o['summary'],coords_s=o['coords_s'],coords_t=o['coords_t'])
    p=spot_p(cfg,pair); layer=LAYER_BY_MAPPING[mapping]; hs,ss,_=natural_assignment(cfg,layer,a); ht,st,_=natural_assignment(cfg,layer,b); return dict(mapping=mapping,pair=pair,p=p,hs=hs,ht=ht,q_direct=natural_q(cfg,mapping,pair),spots_s=ss,spots_t=st,soft_s=hs,soft_t=ht,summary={},coords_s=_coords(h5ad_path(cfg,'spot',a)),coords_t=_coords(h5ad_path(cfg,'spot',b)))

# ---------- standardized core tables ----------
def build_core_tables(cfg: UnifiedSuiteConfig):
    metric_rows=[]; state_rows=[]; spatial_spot_rows=[]; closure_rows=[]
    for pair in cfg.adjacent_pairs:
        pspot=spot_p(cfg,pair); ei_spot=effective_information(pspot); a,b=pair.split('->')
        # Spot-level EI is retained as the spatial micro reference; all macro analyses compare K150/K40/Optimized.
        spot_ids=_obs_index(h5ad_path(cfg,'spot',a)); spot_xy=_coords(h5ad_path(cfg,'spot',a)); spot_contrib=state_ei(pspot)
        for i,(spot,xy,e) in enumerate(zip(spot_ids,spot_xy,spot_contrib)):
            spatial_spot_rows.append(dict(mapping='Spot',time_pair=pair,source_time=a,spot_id=spot,state_index=i,state_ei=float(e),x=float(xy[0]),y=float(xy[1])))
        for mapping in MAPPINGS:
            rec=mapping_record(cfg,mapping,pair); q=rec['q_direct']; de=dict(EI=effective_information(q),H_effect=entropy_prob(q.mean(0)),H_noise=float(np.mean(entropy_rows(q))))
            de['determinism']=np.log2(q.shape[1])-de['H_noise']; de['degeneracy']=np.log2(q.shape[1])-de['H_effect']
            metric_rows.append(dict(mapping=mapping,time_pair=pair,source_time=a,target_time=b,EI=de['EI'],spot_EI=ei_spot,delta_EI_vs_spot=de['EI']-ei_spot,**{k:v for k,v in de.items() if k!='EI'},source_states=q.shape[0],target_states=q.shape[1]))
            sei=state_ei(q); ent=entropy_rows(q)
            for i,(x,e) in enumerate(zip(sei,ent)): state_rows.append(dict(mapping=mapping,time_pair=pair,source_time=a,target_time=b,state_index=i,state=f"M{i+1:03d}",state_ei=float(x),transition_entropy=float(e)))
            labels=np.argmax(rec['hs'],1); coords=rec['coords_s']
            for i,(spot,label,xy) in enumerate(zip(rec['spots_s'],labels,coords)):
                spatial_spot_rows.append(dict(mapping=mapping,time_pair=pair,source_time=a,spot_id=spot,state_index=int(label),state_ei=float(sei[label]),x=float(xy[0]),y=float(xy[1])))
            budget=closure_budget(rec['p'],rec['hs'],rec['ht']); cf=crossfit_error(budget['observed'],budget['source_hard'],budget['weights'],cfg.crossfit_folds,cfg.random_seed)
            pred_sep=row_normalize(budget['source_hard']@q); sep_js=float(np.sum(budget['weights']*js_rows(budget['observed'],pred_sep))); ind_js=float(np.sum(budget['weights']*js_rows(budget['observed'],budget['predicted'])))
            residual_js=js_rows(budget['observed'],budget['predicted'])
            closure_rows.append(dict(mapping=mapping,time_pair=pair,source_time=a,target_time=b,I_available_bits=budget['I_available_bits'],I_retained_bits=budget['I_retained_bits'],closure_leakage_bits=budget['closure_leakage_bits'],closure_quality=budget['closure_quality'],signal_status=budget['signal_status'],induced_mean_js=ind_js,induced_p95_js=float(np.quantile(residual_js,.95)),separately_estimated_mean_js=sep_js,q_operational_gap_js=sep_js-ind_js,active_k_source=budget['source_hard'].shape[1],active_k_target=budget['target_hard'].shape[1],**cf))
    return pd.DataFrame(metric_rows),pd.DataFrame(state_rows),pd.DataFrame(spatial_spot_rows),pd.DataFrame(closure_rows)

# ---------- spatial / effective ----------
def _knn_adjacency(coords,k=6):
    n=len(coords); kk=min(k+1,n); nn=NearestNeighbors(n_neighbors=kk).fit(coords); inds=nn.kneighbors(return_distance=False)
    rows=[]; cols=[]
    for i in range(n):
        for j in inds[i]:
            if j!=i: rows.append(i); cols.append(int(j))
    a=sp.coo_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n)).tocsr(); return a.maximum(a.T)

def spatial_state_metrics(cfg, records_by_pair):
    rows=[]
    for (mapping,pair),rec in records_by_pair.items():
        a,_=pair.split('->'); labels=np.argmax(rec['hs'],1); coords=rec['coords_s']; adj=_knn_adjacency(coords); n=len(labels)
        state_table=state_ei(rec['q_direct'])
        for s in range(rec['hs'].shape[1]):
            idx=np.flatnonzero(labels==s); mask=np.zeros(n,bool); mask[idx]=True; sub=adj[idx][:,idx]
            comp=connected_components(sub,directed=False,return_labels=False) if len(idx)>0 else 0
            # boundary: neighbor edges from member to other labels
            coo=adj[idx].tocoo(); global_cols=coo.col; boundary=np.sum(labels[global_cols]!=s); total=max(len(global_cols),1)
            center=coords[idx].mean(0) if len(idx) else np.zeros(2); radius=float(np.mean(np.linalg.norm(coords[idx]-center,axis=1))) if len(idx) else 0
            scale=max(float(np.linalg.norm(coords.max(0)-coords.min(0))),EPS); radius_norm=radius/scale
            x=mask.astype(float); xbar=x.mean(); W=adj.sum(); den=np.sum((x-xbar)**2); moran=float(n/W*np.sum(adj.multiply(np.outer(x-xbar,x-xbar)))/den) if W>0 and den>EPS else np.nan
            rows.append(dict(mapping=mapping,time_pair=pair,time=a,state_index=s,spot_count=len(idx),connected_components=int(comp),fragmentation=float(comp/max(len(idx),1)),boundary_ratio=float(boundary/total),radius_norm=radius_norm,moran_i=moran,state_ei=float(state_table[s])))
    return pd.DataFrame(rows)

def effective_states(cfg):
    rows=[]
    for ti,time in enumerate(cfg.times):
        for mapping in MAPPINGS:
            if mapping=="Optimized coarse-graining":
                if ti<len(cfg.times)-1:
                    rec=optimized_pair(cfg,f'{time}->{cfg.times[ti+1]}'); s=rec['soft_s']; h=rec['hs']
                else:
                    rec=optimized_pair(cfg,f'{cfg.times[ti-1]}->{time}'); s=rec['soft_t']; h=rec['ht']
                labels=np.argmax(h,1); counts=np.bincount(labels,minlength=h.shape[1]); usage=counts/counts.sum(); # hard effective K primary
                ent=entropy_prob(usage); keff=2**ent; conf=float(np.mean(np.max(s,1))); sent=float(np.mean(entropy_rows(s)))
                nominal=int(rec['summary'].get('K',s.shape[1]))
            else:
                layer=LAYER_BY_MAPPING[mapping]; h,_,_=natural_assignment(cfg,layer,time); labels=np.argmax(h,1); counts=np.bincount(labels,minlength=h.shape[1]); usage=counts/counts.sum(); ent=entropy_prob(usage); keff=2**ent; conf=1.0; sent=0.0; nominal=h.shape[1]
            rows.append(dict(mapping=mapping,time=time,active_k=int(np.sum(counts>0)),nominal_k=nominal,Keff=float(keff),usage_entropy_bits=float(ent),max_usage=float(usage.max()),min_usage=float(usage[usage>0].min()),spot_count=int(counts.sum()),assignment_confidence=conf,assignment_entropy_bits=sent))
    return pd.DataFrame(rows)

# ---------- GRN/CCI mechanism on common spot substrate ----------
def _read_h5ad_expression(path:Path):
    with h5py.File(path,'r') as f:
        def idx(group):
            key=group.attrs['_index']; vals=group[key][:]; return [v.decode() if isinstance(v,(bytes,bytearray)) else str(v) for v in vals]
        units=idx(f['obs']); genes=idx(f['var'])
        node=f['layers/count'] if 'layers' in f and 'count' in f['layers'] else (f['layers/counts'] if 'layers' in f and 'counts' in f['layers'] else f['X'])
        if isinstance(node,h5py.Group):
            shape=tuple(node.attrs['shape']); dat=np.asarray(node['data']); ind=np.asarray(node['indices'],int); ptr=np.asarray(node['indptr'],int); enc=node.attrs.get('encoding-type','csr_matrix'); enc=enc.decode() if isinstance(enc,bytes) else str(enc); mat=(sp.csc_matrix if enc=='csc_matrix' else sp.csr_matrix)((dat,ind,ptr),shape=shape).tocsr()
        else: mat=sp.csr_matrix(np.asarray(node))
    return mat.astype(float),units,genes

def spot_grn_activity(cfg,time):
    mat,units,genes=_read_h5ad_expression(h5ad_path(cfg,'spot',time)); totals=np.asarray(mat.sum(1)).ravel(); mat=sp.diags(1/np.maximum(totals,1))@mat*1e4; mat.data=np.log1p(mat.data)
    g=pd.read_csv(grn_path(cfg,'spot',time),usecols=['regulator','target','weight']); g['regulator']=g.regulator.astype(str); g['target']=g.target.astype(str); g['weight']=pd.to_numeric(g.weight,errors='coerce').abs(); g=g.dropna(); g=g.sort_values(['regulator','weight'],ascending=[True,False]).groupby('regulator',group_keys=False).head(50)
    gi={gene:i for i,gene in enumerate(genes)}; g=g[g.target.isin(gi)].copy(); regs=sorted(g.regulator.unique()); ri={r:i for i,r in enumerate(regs)}; ti=g.target.map(gi).to_numpy(int); rix=g.regulator.map(ri).to_numpy(int); w=g.weight.to_numpy(float); sums=np.bincount(rix,weights=w,minlength=len(regs)); w=w/np.maximum(sums[rix],EPS)
    proj=sp.coo_matrix((w,(ti,rix)),shape=(len(genes),len(regs))).tocsr(); act=(mat@proj).toarray(); return act,units,regs

def mechanism_tables(cfg,records_by_pair,state_table):
    rows=[]
    for pair in cfg.adjacent_pairs:
        source,_=pair.split('->'); act,act_units,regs=spot_grn_activity(cfg,source); unit_lookup={u:i for i,u in enumerate(act_units)}; cci=sp.load_npz(cci_path(cfg,'spot',source)).tocsr().astype(float); idx_units=read_index(index_path(cfg,'spot',source)); cci_lookup={u:i for i,u in enumerate(idx_units)}
        for mapping in MAPPINGS:
            rec=records_by_pair[(mapping,pair)]; order=np.array([unit_lookup[u] for u in rec['spots_s']],int); a=act[order]; h=rec['hs']; sizes=h.sum(0); state_act=(h.T@a)/np.maximum(sizes[:,None],1)
            # CCI reorder then aggregate
            co=np.array([cci_lookup[u] for u in rec['spots_s']],int); c=cci[co][:,co]; cm=(h.T@c@h); cm=np.asarray(cm)
            out=cm.sum(1); inn=cm.sum(0); outp=row_normalize(cm); inp=row_normalize(cm.T); logcap=np.log2(max(cm.shape[0],2)); sei=state_ei(rec['q_direct']); tent=entropy_rows(rec['q_direct'])
            spatial=None
            for s in range(h.shape[1]):
                vals=np.maximum(state_act[s],0); tot=vals.sum(); prob=vals/max(tot,EPS); hh=entropy_prob(prob); conc=1-hh/np.log2(max(len(prob),2)); top10=float(np.sort(prob)[-10:].sum())
                top=';'.join(regs[i] for i in np.argsort(vals)[-5:][::-1])
                rows.append(dict(mapping=mapping,time_pair=pair,source_time=source,state_index=s,state_ei=float(sei[s]),transition_entropy=float(tent[s]),grn_total_activity=float(tot),grn_concentration=float(conc),grn_top10_share=top10,top_regulators=top,cci_out_strength=float(out[s]),cci_in_strength=float(inn[s]),cci_out_log=float(np.log1p(out[s])),cci_in_log=float(np.log1p(inn[s])),cci_out_entropy_norm=float(entropy_prob(outp[s])/logcap),cci_in_entropy_norm=float(entropy_prob(inp[s])/logcap),spot_count=int(sizes[s])))
    return pd.DataFrame(rows)

# ---------- random null ----------
def _shuffle_preserve(labels,rng):
    x=np.asarray(labels).copy(); rng.shuffle(x); return x

def hard_from_labels(labels,k=None):
    labels=np.asarray(labels,int); k=int(labels.max()+1 if k is None else k); h=np.zeros((len(labels),k),float); h[np.arange(len(labels)),labels]=1; return h

def matched_null(cfg,records_by_pair,repeats=None):
    """Matched source-partition null. Target macro definition is fixed, as required for a fair closure null."""
    repeats=cfg.null_repeats if repeats is None else repeats; rng=np.random.default_rng(cfg.random_seed); rows=[]
    for pair in cfg.adjacent_pairs:
        for mapping in MAPPINGS:
            rec=records_by_pair[(mapping,pair)]; base=closure_budget(rec['p'],rec['hs'],rec['ht']); obs=base['observed']; ls=np.argmax(base['source_hard'],1)
            observed_ei=effective_information(base['q_induced'])
            rows.append(dict(mapping=mapping,time_pair=pair,kind='observed',repeat=-1,EI=observed_ei,closure_leakage_bits=base['closure_leakage_bits']))
            for r in range(repeats):
                hs=hard_from_labels(_shuffle_preserve(ls,rng),base['source_hard'].shape[1]); bb=closure_budget_from_observed(obs,hs)
                rows.append(dict(mapping=mapping,time_pair=pair,kind='matched_random',repeat=r,EI=effective_information(bb['q_induced']),closure_leakage_bits=bb['closure_leakage_bits']))
    return pd.DataFrame(rows)

# ---------- multiscale/cross-representation consistency ----------
def _labels_at_time(cfg,mapping,time):
    if mapping=="Optimized coarse-graining":
        i=cfg.times.index(time)
        if i<len(cfg.times)-1: return np.argmax(optimized_pair(cfg,f'{time}->{cfg.times[i+1]}')['hs'],1)
        return np.argmax(optimized_pair(cfg,f'{cfg.times[i-1]}->{time}')['ht'],1)
    return np.argmax(natural_assignment(cfg,LAYER_BY_MAPPING[mapping],time)[0],1)

def consistency_table(cfg,closure_table):
    rows=[]
    for time in cfg.times:
        lab={m:_labels_at_time(cfg,m,time) for m in MAPPINGS}
        for i,m1 in enumerate(MAPPINGS):
            for m2 in MAPPINGS[i+1:]:
                rows.append(dict(time=time,mapping_a=m1,mapping_b=m2,NMI=normalized_mutual_info_score(lab[m1],lab[m2]),ARI=adjusted_rand_score(lab[m1],lab[m2])))
    # optimized shared-time incoming/outgoing stability
    for time in cfg.times[1:-1]:
        i=cfg.times.index(time); prev=optimized_pair(cfg,f'{cfg.times[i-1]}->{time}'); nxt=optimized_pair(cfg,f'{time}->{cfg.times[i+1]}')
        # align by spot id
        lp={u:j for j,u in enumerate(prev['spots_t'])}; common=[u for u in nxt['spots_s'] if u in lp]; la=np.array([np.argmax(prev['ht'][lp[u]]) for u in common]); lb=np.array([np.argmax(nxt['hs'][nxt['spots_s'].index(u)]) for u in common])
        rows.append(dict(time=time,mapping_a='Optimized incoming',mapping_b='Optimized outgoing',NMI=normalized_mutual_info_score(la,lb),ARI=adjusted_rand_score(la,lb)))
    return pd.DataFrame(rows)

# ---------- fate paths ----------
def _alignment_hard(spots_a,h_a,spots_b,h_b):
    ia={u:i for i,u in enumerate(spots_a)}; ib={u:i for i,u in enumerate(spots_b)}; common=sorted(set(ia)&set(ib)); A=np.zeros((h_a.shape[1],h_b.shape[1]))
    for u in common: A[np.argmax(h_a[ia[u]]),np.argmax(h_b[ib[u]])]+=1
    return row_normalize(A),len(common)

def mapping_q_chain(cfg,mapping):
    recs=[mapping_record(cfg,mapping,p) for p in cfg.adjacent_pairs]
    qs=[]; align=[]
    for r in recs: qs.append(r['q_direct'])
    for i in range(len(recs)-1):
        if mapping=="Optimized coarse-graining": A,_=_alignment_hard(recs[i]['spots_t'],recs[i]['ht'],recs[i+1]['spots_s'],recs[i+1]['hs'])
        else: A=np.eye(qs[i].shape[1],qs[i+1].shape[0])
        align.append(A)
    return recs,qs,align

def fate_table(cfg):
    rows=[]
    for mapping in MAPPINGS:
        recs,qs,al=mapping_q_chain(cfg,mapping); q01,q12,q23=qs; A1,A2=al; comp=row_normalize(q01@A1@q12@A2@q23); sei=state_ei(q01); bentr=entropy_rows(q01)
        for s in range(q01.shape[0]):
            s1=int(np.argmax(q01[s])); v1=q01[s,s1]; # align target1 to source1 distribution
            mid1=A1[s1]; s1b=int(np.argmax(mid1)); vA1=mid1[s1b]; s2=int(np.argmax(q12[s1b])); v2=q12[s1b,s2]; mid2=A2[s2]; s2b=int(np.argmax(mid2)); vA2=mid2[s2b]; s3=int(np.argmax(q23[s2b])); v3=q23[s2b,s3]
            rows.append(dict(mapping=mapping,state_t0=s,state_t1=s1,state_t2=s2,state_t3=s3,path_probability=float(v1*vA1*v2*vA2*v3),endpoint_entropy=float(entropy_rows(comp[s:s+1])[0]),source_ei=float(sei[s]),first_branch_entropy=float(bentr[s])))
    return pd.DataFrame(rows)

# ---------- perturbation ----------
def _blend_rows(q,idx,dose):
    q=row_normalize(q).copy(); mean=q.mean(0); q[idx]=(1-dose)*q[idx]+dose*mean; return row_normalize(q)
def perturbation_table(cfg,records_by_pair,mechanism):
    rng=np.random.default_rng(cfg.random_seed+17); rows=[]; doses=np.linspace(0,1,6)
    for pair in cfg.adjacent_pairs:
        for mapping in MAPPINGS:
            rec=records_by_pair[(mapping,pair)]; q=rec['q_direct']; mech=mechanism[(mechanism.mapping==mapping)&(mechanism.time_pair==pair)].set_index('state_index'); n=q.shape[0]; k=max(1,int(round(.2*n))); sei=state_ei(q); sels={'high_state_ei':np.argsort(sei)[-k:], 'high_cci_out':np.argsort(mech.cci_out_log.reindex(range(n)).to_numpy())[-k:], 'high_grn_concentration':np.argsort(mech.grn_concentration.reindex(range(n)).to_numpy())[-k:]}; base=effective_information(q)
            for target,idx in sels.items():
                for dose in doses: rows.append(dict(mapping=mapping,time_pair=pair,target=target,dose=dose,ei_drop_mean=base-effective_information(_blend_rows(q,idx,dose)),ei_drop_low=np.nan,ei_drop_high=np.nan))
            for dose in doses:
                drops=[]
                for _ in range(cfg.perturb_random_repeats): idx=rng.choice(n,size=k,replace=False); drops.append(base-effective_information(_blend_rows(q,idx,dose)))
                rows.append(dict(mapping=mapping,time_pair=pair,target='matched_random',dose=dose,ei_drop_mean=np.mean(drops),ei_drop_low=np.quantile(drops,.025),ei_drop_high=np.quantile(drops,.975)))
    return pd.DataFrame(rows)

# ---------- horizon / CK ----------
def horizon_table(cfg):
    rows=[]
    ps=[spot_p(cfg,p) for p in cfg.adjacent_pairs]
    for mapping in MAPPINGS:
        # endpoints: natural assignments global; optimized use pair endpoints
        for start in range(0,3):
            pcomp=np.eye(ps[start].shape[0])
            for end in range(start,3):
                pcomp=row_normalize(pcomp@ps[end]); hsteps=end-start+1; t0=cfg.times[start]; t1=cfg.times[end+1]
                if mapping=="Optimized coarse-graining":
                    hs=optimized_pair(cfg,cfg.adjacent_pairs[start])['hs']; ht=optimized_pair(cfg,cfg.adjacent_pairs[end])['ht']
                else:
                    layer=LAYER_BY_MAPPING[mapping]; hs=natural_assignment(cfg,layer,t0)[0]; ht=natural_assignment(cfg,layer,t1)[0]
                b=closure_budget(pcomp,hs,ht)
                # composed separately estimated Q
                recs=[mapping_record(cfg,mapping,cfg.adjacent_pairs[i]) for i in range(start,end+1)]; qcomp=recs[0]['q_direct']
                for j in range(len(recs)-1):
                    if mapping=="Optimized coarse-graining": A,_=_alignment_hard(recs[j]['spots_t'],recs[j]['ht'],recs[j+1]['spots_s'],recs[j+1]['hs'])
                    else: A=np.eye(qcomp.shape[1],recs[j+1]['q_direct'].shape[0])
                    qcomp=row_normalize(qcomp@A@recs[j+1]['q_direct'])
                pred=row_normalize(b['source_hard']@qcomp); sep=float(np.sum(b['weights']*js_rows(b['observed'],pred))); ind=float(np.sum(b['weights']*js_rows(b['observed'],b['predicted'])))
                rows.append(dict(mapping=mapping,start_time=t0,end_time=t1,horizon=hsteps,time_pair=f'{t0}->{t1}',I_available_bits=b['I_available_bits'],I_retained_bits=b['I_retained_bits'],closure_leakage_bits=b['closure_leakage_bits'],closure_quality=b['closure_quality'],signal_status=b['signal_status'],ck_composition_excess_js=sep-ind))
    return pd.DataFrame(rows)

# ---------- repairability ----------
def leakage_state_table(cfg,records_by_pair):
    rows=[]; spotrows=[]
    for (mapping,pair),rec in records_by_pair.items():
        b=closure_budget(rec['p'],rec['hs'],rec['ht']); labels=np.argmax(b['source_hard'],1); kres=kl_rows(b['observed'],b['predicted']); total=b['closure_leakage_bits']; coords=rec['coords_s']
        for s in range(b['source_hard'].shape[1]):
            idx=labels==s; leak=float(np.sum(b['weights'][idx]*kres[idx])); rows.append(dict(mapping=mapping,time_pair=pair,state_index=s,spot_count=int(idx.sum()),leakage_bits=leak,leakage_share=leak/max(total,EPS),p95_js=float(np.quantile(js_rows(b['observed'][idx],b['predicted'][idx]),.95))))
        for i,(u,c) in enumerate(zip(rec['spots_s'],coords)): spotrows.append(dict(mapping=mapping,time_pair=pair,spot_id=u,x=float(c[0]),y=float(c[1]),spot_leakage_kl=float(b['weights'][i]*kres[i]),state_index=int(labels[i])))
    return pd.DataFrame(rows),pd.DataFrame(spotrows)

def _binary_split_future(obs,idx,seed):
    from sklearn.cluster import KMeans
    if len(idx)<2:return None
    labels=KMeans(n_clusters=2,n_init=10,random_state=seed).fit_predict(obs[idx]);
    if len(np.unique(labels))<2:return None
    return labels

def repairability_table(cfg,records_by_pair):
    """Oracle headroom using exactly the primary hard-target/state-balanced KL leakage semantics."""
    from sklearn.cluster import KMeans
    rng=np.random.default_rng(cfg.random_seed+29); rows=[]
    for pair in cfg.adjacent_pairs:
        for mapping in MAPPINGS:
            rec=records_by_pair[(mapping,pair)]; base=closure_budget(rec['p'],rec['hs'],rec['ht']); obs=base['observed']; current=np.argmax(base['source_hard'],1).copy()
            rows.append(dict(mapping=mapping,time_pair=pair,step=0,leakage_bits=base['closure_leakage_bits'],targeted_gain=0.0,random_gain_mean=0.0,gain_over_random=np.nan,active_k=len(np.unique(current))))
            for step in range(1,cfg.repair_max_splits+1):
                hs=hard_from_labels(current); b=closure_budget_from_observed(obs,hs); lbl=np.argmax(b['source_hard'],1); kres=kl_rows(b['observed'],b['predicted']); candidates=[]
                for state in np.unique(lbl):
                    idx=np.flatnonzero(lbl==state)
                    if len(idx)>=4: candidates.append((float(np.sum(b['weights'][idx]*kres[idx])),state,idx))
                if not candidates: break
                _,state,idx=max(candidates); km=KMeans(n_clusters=2,random_state=cfg.random_seed+step,n_init=10).fit(obs[idx]); split=km.labels_; child=int(np.sum(split==1))
                if child==0 or child==len(idx): break
                new=current.copy(); new_label=int(current.max()+1); new[idx[split==1]]=new_label; after=closure_budget_from_observed(obs,hard_from_labels(new))['closure_leakage_bits']; gain=b['closure_leakage_bits']-after
                rg=[]
                for _ in range(cfg.repair_random_repeats):
                    sel=rng.choice(idx,size=child,replace=False); rr=current.copy(); rr[sel]=new_label; rg.append(b['closure_leakage_bits']-closure_budget_from_observed(obs,hard_from_labels(rr))['closure_leakage_bits'])
                rmean=float(np.mean(rg)); current=new; rows.append(dict(mapping=mapping,time_pair=pair,step=step,leakage_bits=after,targeted_gain=gain,random_gain_mean=rmean,gain_over_random=gain/max(rmean,EPS) if rmean>0 else np.inf,active_k=len(np.unique(current))))
    return pd.DataFrame(rows)

# ---------- robustness ----------
def robustness_tables(cfg,records_by_pair,closure_table,null_table):
    weight_rows=[]; soft_rows=[]; boot_rows=[]; rng=np.random.default_rng(cfg.random_seed+7)
    for (mapping,pair),rec in records_by_pair.items():
        for weight in ('state_balanced','uniform_spot'):
            # custom budget weighting
            p=rec['p']; hs,_=hard_compact(rec['hs']); ht,_=hard_compact(rec['ht']); obs=row_normalize(p@ht); w=state_balanced_weights(hs) if weight=='state_balanced' else np.full(len(hs),1/len(hs)); q,m=induced_q(obs,hs,w); pred=row_normalize(hs@q); leak=float(np.sum(w*kl_rows(obs,pred))); avail=weighted_information(obs,w); ret=weighted_information(q,m)
            weight_rows.append(dict(mapping=mapping,time_pair=pair,weighting=weight,closure_leakage_bits=leak,I_available_bits=avail,I_retained_bits=ret))
        # bootstrap source microstates within hard macro; future hard-macro profiles are fixed
        base_boot=closure_budget(rec['p'],rec['hs'],rec['ht']); obs_boot=base_boot['observed']; hs_boot=base_boot['source_hard']; labels=np.argmax(hs_boot,1); vals=[]
        for _ in range(cfg.bootstrap_repeats):
            selected=[]
            for state in np.unique(labels):
                idx=np.flatnonzero(labels==state); selected.extend(rng.choice(idx,size=len(idx),replace=True).tolist())
            selected=np.array(selected,int); vals.append(closure_budget_from_observed(obs_boot[selected],hs_boot[selected])['closure_leakage_bits'])
        boot_rows.append(dict(mapping=mapping,time_pair=pair,mean=float(np.mean(vals)),low=float(np.quantile(vals,.025)),high=float(np.quantile(vals,.975))))
        if mapping=="Optimized coarse-graining":
            # soft-coordinate bounded diagnostic only
            p=rec['p']; ss=rec['soft_s']; st=rec['soft_t']; obs=row_normalize(p@st); w=np.full(len(ss),1/len(ss)); q,m=induced_q(obs,ss,w); pred=row_normalize(ss@q); soft_js=float(np.mean(js_rows(obs,pred))); hard=float(closure_table[(closure_table.mapping==mapping)&(closure_table.time_pair==pair)].induced_mean_js.iloc[0])
            soft_rows.append(dict(mapping=mapping,time_pair=pair,hard_primary_js=hard,soft_coordinate_js=soft_js,mean_confidence=float(np.mean(np.max(ss,1))),mean_assignment_entropy=float(np.mean(entropy_rows(ss)))))
    return pd.DataFrame(weight_rows),pd.DataFrame(boot_rows),pd.DataFrame(soft_rows)

# ---------- candidate Pareto sweep ----------
def candidate_sweep(cfg: UnifiedSuiteConfig, records_by_pair):
    out=Path(cfg.output_root)/'candidate_sweep'; out.mkdir(parents=True,exist_ok=True); rows=[]
    # canonical anchors: all three primary representations use the same EI-vs-spot definition
    for pair in cfg.adjacent_pairs:
        ei_sp=effective_information(spot_p(cfg,pair))
        for mapping in MAPPINGS:
            rec=records_by_pair[(mapping,pair)]; b=closure_budget(rec['p'],rec['hs'],rec['ht']); cf=crossfit_error(b['observed'],b['source_hard'],b['weights'],cfg.crossfit_folds,cfg.random_seed); pred=row_normalize(b['source_hard']@rec['q_direct']); sep=float(np.sum(b['weights']*js_rows(b['observed'],pred))); ind=float(np.sum(b['weights']*js_rows(b['observed'],b['predicted']))); ak=b['source_hard'].shape[1]
            nominal=ak; conf=1.0; sent=0.0
            if mapping=="Optimized coarse-graining":
                nominal=int(rec.get('summary',{}).get('K',rec['soft_s'].shape[1])); conf=float(np.mean(np.max(row_normalize(rec['soft_s']),1))); sent=float(np.mean(entropy_rows(row_normalize(rec['soft_s']))))
            rows.append(dict(candidate_id=mapping,mapping=mapping,time_pair=pair,nominal_k=nominal,active_k=ak,seed=-1,canonical_delta_ei=effective_information(rec['q_direct'])-ei_sp,I_available_bits=b['I_available_bits'],I_retained_bits=b['I_retained_bits'],closure_leakage_bits=b['closure_leakage_bits'],closure_quality=b['closure_quality'],signal_status=b['signal_status'],crossfit_kl_bits=cf['crossfit_kl_bits'],crossfit_coverage=cf['crossfit_coverage'],q_operational_gap_js=sep-ind,compression_gain=1-np.log2(max(ak,1))/np.log2(rec['p'].shape[0]),assignment_confidence=conf,assignment_entropy=sent))
    if cfg.run_candidate_sweep:
        from mignet_ce.visualization.downstream.dynamic_closure.optimal import OptimalRunConfig, prepare_optimal_input
        from wyt_deltaei_coarse_grain import WYTDeltaEIConfig, train_deltaei
        for pair in cfg.adjacent_pairs:
            a,btime=pair.split('->'); prep_cfg=OptimalRunConfig(data_root=Path(cfg.data_root),output_root=out/'_prepared',k=40,epochs=1,nmf_components=5,nmf_max_iter=cfg.candidate_nmf_max_iter,large_target_nmf_max_iter=cfg.candidate_large_nmf_max_iter,seed=42,device='cpu')
            prepared=prepare_optimal_input(prep_cfg,a,btime); ei_sp=effective_information(prepared.micro_pij)
            for k in cfg.candidate_k_grid:
                for seed in cfg.candidate_seeds:
                    d=out/pair.replace('->','_to_')/f'K{k}_seed{seed}'; summary=d/'summary.json'
                    if not summary.exists():
                        train_deltaei(prepared,WYTDeltaEIConfig(k=int(k),out_dir=d,epochs=cfg.candidate_epochs,seed=int(seed),device='cpu',log_every=max(10,cfg.candidate_epochs)))
                    st=np.load(d/'S_t.npy'); sp_=np.load(d/'S_tp.npy'); hs,asrc=hard_compact(st); ht,atgt=hard_compact(sp_); q=row_normalize(np.load(d/'PIJ_macro_train.npy'))[np.ix_(asrc,atgt)]; q=row_normalize(q); bgt=closure_budget(prepared.micro_pij,hs,ht); cf=crossfit_error(bgt['observed'],bgt['source_hard'],bgt['weights'],cfg.crossfit_folds,cfg.random_seed+seed); pred=row_normalize(bgt['source_hard']@q); sep=float(np.sum(bgt['weights']*js_rows(bgt['observed'],pred))); ind=float(np.sum(bgt['weights']*js_rows(bgt['observed'],bgt['predicted']))); ak=hs.shape[1]
                    rows.append(dict(candidate_id=f'Opt K{k} s{seed}',mapping='Optimized candidate',time_pair=pair,nominal_k=k,active_k=ak,seed=seed,canonical_delta_ei=effective_information(q)-ei_sp,I_available_bits=bgt['I_available_bits'],I_retained_bits=bgt['I_retained_bits'],closure_leakage_bits=bgt['closure_leakage_bits'],closure_quality=bgt['closure_quality'],signal_status=bgt['signal_status'],crossfit_kl_bits=cf['crossfit_kl_bits'],crossfit_coverage=cf['crossfit_coverage'],q_operational_gap_js=sep-ind,compression_gain=1-np.log2(max(ak,1))/np.log2(prepared.micro_pij.shape[0]),assignment_confidence=float(np.mean(np.max(row_normalize(st),1))),assignment_entropy=float(np.mean(entropy_rows(row_normalize(st))))))
    frame=pd.DataFrame(rows)
    # 2D front flags within each pair
    frame['front_emergence_closure']=False; frame['front_compression_prediction']=False
    for pair,g in frame.groupby('time_pair'):
        ids=g.index.tolist()
        for i in ids:
            a=frame.loc[i];
            if a.signal_status!='informative' or not np.isfinite(a.closure_quality): continue
            dom=False
            for j in ids:
                if i==j:continue
                b=frame.loc[j]
                if b.signal_status!='informative' or not np.isfinite(b.closure_quality):continue
                if b.canonical_delta_ei>=a.canonical_delta_ei-EPS and b.closure_quality>=a.closure_quality-EPS and (b.canonical_delta_ei>a.canonical_delta_ei+EPS or b.closure_quality>a.closure_quality+EPS): dom=True;break
            if not dom: frame.loc[i,'front_emergence_closure']=True
            dom=False
            for j in ids:
                if i==j:continue
                b=frame.loc[j]
                if b.compression_gain>=a.compression_gain-EPS and b.I_retained_bits>=a.I_retained_bits-EPS and (b.compression_gain>a.compression_gain+EPS or b.I_retained_bits>a.I_retained_bits+EPS): dom=True;break
            if not dom: frame.loc[i,'front_compression_prediction']=True
    return frame

# ---------- orchestrator ----------
def build_all_tables(cfg: UnifiedSuiteConfig):
    metrics,states,spatial_spots,closure=build_core_tables(cfg); records={(m,p):mapping_record(cfg,m,p) for p in cfg.adjacent_pairs for m in MAPPINGS}
    spatial=spatial_state_metrics(cfg,records); effective=effective_states(cfg); mechanism=mechanism_tables(cfg,records,states); null=matched_null(cfg,records); consistency=consistency_table(cfg,closure); fate=fate_table(cfg); perturb=perturbation_table(cfg,records,mechanism); horizon=horizon_table(cfg); leak_states,leak_spots=leakage_state_table(cfg,records); repair=repairability_table(cfg,records); weights,boot,soft=robustness_tables(cfg,records,closure,null); pareto=candidate_sweep(cfg,records)
    tables=dict(metrics=metrics,states=states,spatial_spots=spatial_spots,closure=closure,spatial=spatial,effective=effective,mechanism=mechanism,null=null,consistency=consistency,fate=fate,perturbation=perturb,horizon=horizon,leakage_states=leak_states,leakage_spots=leak_spots,repairability=repair,weighting=weights,bootstrap=boot,soft_hard=soft,pareto=pareto)
    return tables,records

def save_tables(tables:dict[str,pd.DataFrame],root:Path):
    tdir=Path(root)/'tables'; tdir.mkdir(parents=True,exist_ok=True)
    for name,df in tables.items(): df.to_csv(tdir/f'{name}.csv',index=False)
    return tdir
