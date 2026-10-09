"""Formal ablation: direct production feature/PIJ kernels, no legacy ablation registry.

Uses production low-level NMF/Lap features with per-pair z-score, and the
integrated production GRNPairEnhancer. No method selection performed.
"""
from __future__ import annotations
import numpy as np
from mignet_ce.config import TemporalRunConfig, VerticalPairSpec
from mignet_ce.io.loaders import LayerDataResolver
from mignet_ce.metrics import effective_information, pairwise_joint_nmf, pairwise_shared_core_directed_nmf
from mignet_ce.networks.registry import get_network_builder
from mignet_ce.pij.compare._shared.features import (
    adjacency_from_lightcci_graph, laplacian_hks_features,
    _build_pairwise_grn_state_side, _standardize_pairwise_features,
)
from mignet_ce.pij.compare._shared.kl import pairwise_feature_kl
from mignet_ce.pij.compare._shared.cosine import row_normalized_kernel_from_cost
from mignet_ce.pij.compare._shared.ng_kl_ot import canonical_ng_pij_from_cost_numpy
from .config import (FormalAblationConfig, TIME_POINTS, TIME_PAIRS,
                     T1_CONFIGS, FIXED_GRN, FIXED_CCI, GRN_PROVIDER)


def make_temporal(cfg: FormalAblationConfig, hierarchy: tuple[str,str,str], grn: str) -> TemporalRunConfig:
    _,lo,hi = hierarchy
    t = TemporalRunConfig(
        data_root=cfg.data_root, output_root=cfg.output_root, organs=(cfg.organ,),
        time_points=TIME_POINTS, level_pairs=(VerticalPairSpec(lo,hi),),
        network_method='light_cci_grn', pij_method='NG_KLot',
        embedding_method='joint_nmf', grn_feature_method=GRN_PROVIDER[grn],
        grn_residual_lambda=cfg.grn_residual_lambda,
        nmf_components=cfg.nmf_components, nmf_max_iter=cfg.nmf_max_iter,
        nmf_seed=cfg.nmf_seed, laplacian_components=cfg.laplacian_components,
        pij_entropy_epsilon=cfg.beta, export_features=False, export_graphs=False,
        export_pair_artifacts=False, export_feature_diagnostics=False,
    )
    t.validate()
    return t

def load_context(cfg: FormalAblationConfig, hierarchy: tuple[str,str,str]):
    temporal = make_temporal(cfg,hierarchy,'NMF')
    resolver = LayerDataResolver(cfg.data_root)
    # Production builder enforces CCI index alignment; absence is a hard failure.
    return get_network_builder('light_cci_grn').build_pair_context(
        organ=cfg.organ, pair=temporal.level_pairs[0], cfg=temporal, resolver=resolver)

def _cci_pair(context, cfg, side, pair, method):
    """Original CCI feature extraction, pairwise source/target z-score.

    Uses the same production low-level NMF/HKS functions as the Round-11 pilot.
    This matters: the generic compare feature builder standardizes HKS across
    all timepoints, whereas the published pilot uses per-time-pair scaling.
    """
    graphs=context.lower_graphs if side=='lower' else context.upper_graphs
    units=context.lower_units_by_time if side=='lower' else context.upper_units_by_time
    mats=[]
    for idx in pair:
        a, meta=adjacency_from_lightcci_graph(graphs[idx],units[idx],cci_min=0.0)
        if int(meta.get('requested_units',len(units[idx]))) != len(units[idx]):
            raise ValueError('CCI unit count failed alignment validation')
        mats.append(a)
    if method=='NMF':
        if (context.pair.lower_layer if side=='lower' else context.pair.upper_layer)=='spot':
            u,v,x,y,_=pairwise_shared_core_directed_nmf(
                mats[0].toarray(),mats[1].toarray(),
                n_components=cfg.nmf_components,max_iter=cfg.nmf_max_iter,seed=cfg.nmf_seed)
            features=(np.column_stack([u,v]),np.column_stack([x,y]))
        else:
            u,v,_=pairwise_joint_nmf(mats[0].toarray(),mats[1].toarray(),
                      n_components=cfg.nmf_components,max_iter=cfg.nmf_max_iter,seed=cfg.nmf_seed)
            features=(u,v)
    else:
        features=tuple(laplacian_hks_features(a,n_components=cfg.laplacian_components,normalized=True) for a in mats)
    standardized,_=_standardize_pairwise_features({pair:features},context)
    s,t=standardized[pair]
    if s.shape[1]!=t.shape[1] or not np.isfinite(s).all() or not np.isfinite(t).all():
        raise RuntimeError(f'Invalid {side} {method} features for pair {pair}')
    return pairwise_feature_kl(s,t,beta=cfg.beta)

def build_cost_blocks(cfg: FormalAblationConfig, hierarchy: tuple[str,str,str], context) -> dict:
    """GRN via integrated production GRN feature path, CCI via original low-level NMF/Lap."""
    costs={}
    for side in ('lower','upper'):
        for grn in ('NMF','Laplacian'):
            tmpcfg=make_temporal(cfg,hierarchy,grn)
            pair_indices=tuple((TIME_POINTS.index(p.split('->')[0]),TIME_POINTS.index(p.split('->')[1])) for p in TIME_PAIRS)
            raw,_=_build_pairwise_grn_state_side(context,side,pair_indices,tmpcfg)
            if raw is None:raise RuntimeError(f'GRN features unavailable for {side} in {hierarchy[0]}')
            for label,pair in zip(TIME_PAIRS,pair_indices):
                s,t=raw[pair]
                costs[(side,label,'GRN',grn)] = pairwise_feature_kl(s,t,beta=cfg.beta)
        for label in TIME_PAIRS:
            pair=(TIME_POINTS.index(label.split('->')[0]),TIME_POINTS.index(label.split('->')[1]))
            for cci in ('NMF','Laplacian'):
                costs[(side,label,'CCI',cci)]=_cci_pair(context,cfg,side,pair,cci)
    return costs

def experiment_specs() -> tuple[tuple[str,str,str,str,str], ...]:
    # (table, GRN feature, CCI feature, PIJ, network input)
    return tuple([('table1',g,c,'KL+OT','GRN+CCI') for g,c in T1_CONFIGS] + [
        ('table2',FIXED_GRN,FIXED_CCI,'KL','GRN+CCI'),
        ('table2',FIXED_GRN,FIXED_CCI,'KL+OT','GRN+CCI'),
        ('table3',FIXED_GRN,FIXED_CCI,'KL+OT','GRN-only'),
        ('table3',FIXED_GRN,FIXED_CCI,'KL+OT','CCI-only'),
        ('table3',FIXED_GRN,FIXED_CCI,'KL+OT','GRN+CCI'),
    ])

def transition(cost: np.ndarray, method: str, tau: float):
    if cost.ndim != 2 or cost.size == 0 or not np.isfinite(cost).all() or (cost < -1e-10).any():
        raise ValueError('Invalid formal-ablation raw KL cost.')
    if method == 'KL':
        _,p = row_normalized_kernel_from_cost(cost,tau=tau)
        return p, {'ot_residual': np.nan, 'ot_iterations': 0, 'ot_fallback': False}
    if method == 'KL+OT':
        _,p,info = canonical_ng_pij_from_cost_numpy(cost,temperature=tau)
        d = info.get('sinkhorn',{})
        residual = d.get('max_absolute_marginal_residual',d.get('marginal_residual',np.nan))
        return p, {'ot_residual': float(residual),
                   'ot_iterations': int(d.get('iterations',0)),
                   'ot_fallback': bool(d.get('log_domain_fallback_used',False))}
    raise ValueError(method)

def evaluate_cost_blocks(cfg: FormalAblationConfig, hierarchy: tuple[str,str,str],
                         costs: dict) -> list[dict]:
    label,lower,upper = hierarchy
    rows = []
    # Deduplicate cross-table conditions, but retain each table's intended output rows.
    transition_cache = {}
    for table,grn,cci,pij,input_group in experiment_specs():
        for time_pair in TIME_PAIRS:
            result = {}
            for side,layer in [('lower',lower),('upper',upper)]:
                dg = costs[(side,time_pair,'GRN',grn)]
                dc = costs[(side,time_pair,'CCI',cci)]
                if dg.shape != dc.shape: raise ValueError(f'GRN/CCI cost shape mismatch: {dg.shape} and {dc.shape}')
                if input_group == 'GRN-only': combined = dg
                elif input_group == 'CCI-only': combined = dc
                else: combined = (1.0-cfg.alpha_cci)*dg+cfg.alpha_cci*dc
                key = (side,time_pair,grn,cci,pij,input_group)
                if key not in transition_cache:
                    matrix, diagnostics = transition(combined,pij,cfg.temperature)
                    if not np.isfinite(matrix).all() or np.max(abs(matrix.sum(axis=1)-1))>1e-5:
                        raise RuntimeError('PIJ is not finite row-stochastic')
                    transition_cache[key] = (float(effective_information(matrix.copy())),diagnostics,combined.shape)
                result[side] = transition_cache[key]
            eli, dlo, _ = result['lower']; eui, dup, _ = result['upper']
            rows.append(dict(table=table,hierarchy=label,lower_layer=lower,upper_layer=upper,
                             time_pair=time_pair,grn_feature=grn,cci_feature=cci,
                             pij=pij,input=input_group,alpha=cfg.alpha_cci if input_group=='GRN+CCI' else 0.0 if input_group=='GRN-only' else 1.0,
                             EI_lower=eli,EI_upper=eui,delta_EI=eui-eli,
                             ot_residual_lower=dlo['ot_residual'],ot_residual_upper=dup['ot_residual'],
                             ot_iterations_lower=dlo['ot_iterations'],ot_iterations_upper=dup['ot_iterations'],
                             ot_fallback_lower=dlo['ot_fallback'],ot_fallback_upper=dup['ot_fallback'],
                             n_source_lower=result['lower'][2][0],n_target_lower=result['lower'][2][1],
                             n_source_upper=result['upper'][2][0],n_target_upper=result['upper'][2][1]))
    return rows
