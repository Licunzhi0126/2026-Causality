from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp

from .analysis import load_domain_assignment
from .config import DEFAULT_TIME_POINTS
from .io import layer_paths, read_index, write_manifest
from .ultradeep_plots import render_ultradeep_figures

EPS = 1e-12
TIMES = DEFAULT_TIME_POINTS
PAIRS = tuple(f"{a}->{b}" for a, b in zip(TIMES[:-1], TIMES[1:]))
TRIPLES = tuple(zip(TIMES[:-2], TIMES[1:-1], TIMES[2:]))
MAPPINGS = ("Seurat K150", "Seurat K40", "Optimized coarse-graining")


def row_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.maximum(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    s = x.sum(axis=1, keepdims=True)
    return np.divide(x, s, out=np.full_like(x, 1.0 / max(x.shape[1], 1)), where=s > EPS)


def normalize_assignment(s: np.ndarray) -> np.ndarray:
    return row_normalize(s)


def load_matrix(path: Path) -> np.ndarray:
    path = Path(path)
    try:
        return np.asarray(sp.load_npz(path).toarray(), dtype=np.float64)
    except Exception:
        z = np.load(path)
        if isinstance(z, np.ndarray):
            return np.asarray(z, dtype=np.float64)
        if "pij" in z.files:
            return np.asarray(z["pij"], dtype=np.float64)
        if len(z.files) == 1:
            return np.asarray(z[z.files[0]], dtype=np.float64)
        raise ValueError(f"Cannot infer matrix in {path}; keys={z.files}")


def entropy_prob(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = p[p > EPS]
    return float(-np.sum(p * np.log2(p)))


def entropy_rows(p: np.ndarray) -> np.ndarray:
    p = row_normalize(p)
    return -np.sum(np.where(p > EPS, p * np.log2(np.maximum(p, EPS)), 0.0), axis=1)


def js_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = row_normalize(p); q = row_normalize(q); m = 0.5 * (p + q)
    kl1 = np.sum(np.where(p > EPS, p * np.log2(np.maximum(p, EPS) / np.maximum(m, EPS)), 0.0), axis=1)
    kl2 = np.sum(np.where(q > EPS, q * np.log2(np.maximum(q, EPS) / np.maximum(m, EPS)), 0.0), axis=1)
    return 0.5 * (kl1 + kl2)


def mi_joint(joint: np.ndarray) -> float:
    joint = np.maximum(np.asarray(joint, dtype=np.float64), 0.0)
    joint /= max(float(joint.sum()), EPS)
    px = joint.sum(axis=1, keepdims=True); py = joint.sum(axis=0, keepdims=True)
    denom = px @ py
    mask = joint > EPS
    return float(np.sum(joint[mask] * np.log2(joint[mask] / np.maximum(denom[mask], EPS))))


def conditional_mi_triplet(joint: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return I(A;C|B), per-B conditional MI, and P(B)."""
    p = np.maximum(np.asarray(joint, dtype=np.float64), 0.0)
    p /= max(float(p.sum()), EPS)
    pab = p.sum(axis=2)          # a,b
    pbc = p.sum(axis=0)          # b,c
    pb = pbc.sum(axis=1)         # b
    cmi = 0.0
    per_b = np.zeros(p.shape[1], dtype=float)
    for b in range(p.shape[1]):
        if pb[b] <= EPS:
            continue
        pac_b = p[:, b, :] / pb[b]
        val = mi_joint(pac_b)
        per_b[b] = val
        cmi += pb[b] * val
    return float(cmi), per_b, pb


def conditional_entropy_c_given_b(joint: np.ndarray) -> float:
    p = np.maximum(np.asarray(joint, dtype=np.float64), 0.0)
    p /= max(float(p.sum()), EPS)
    pbc = p.sum(axis=0)
    pb = pbc.sum(axis=1)
    h = 0.0
    for b in range(len(pb)):
        if pb[b] > EPS:
            h += pb[b] * entropy_prob(pbc[b] / pb[b])
    return float(h)


def conditional_entropy_c_given_ab(joint: np.ndarray) -> float:
    p = np.maximum(np.asarray(joint, dtype=np.float64), 0.0)
    p /= max(float(p.sum()), EPS)
    pab = p.sum(axis=2)
    h = 0.0
    for a in range(p.shape[0]):
        for b in range(p.shape[1]):
            if pab[a,b] > EPS:
                h += pab[a,b] * entropy_prob(p[a,b,:] / pab[a,b])
    return float(h)


def macro_triplet_joint(p01: np.ndarray, p12: np.ndarray, s0: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Joint macro path distribution induced by uniform interventions on t0 microstates.

    J[a,b,c] = sum_i pi_i S0[i,a] sum_j P01[i,j] S1[j,b]
               sum_k P12[j,k] S2[k,c].
    """
    p01 = row_normalize(p01); p12 = row_normalize(p12)
    s0 = normalize_assignment(s0); s1 = normalize_assignment(s1); s2 = normalize_assignment(s2)
    if p01.shape != (s0.shape[0], s1.shape[0]) or p12.shape != (s1.shape[0], s2.shape[0]):
        raise ValueError(f"Triplet shapes incompatible: {p01.shape}, {p12.shape}, {s0.shape}, {s1.shape}, {s2.shape}")
    pi = np.full(s0.shape[0], 1.0 / s0.shape[0], dtype=np.float64)
    w = (s0 * pi[:, None]).T @ p01                # K0 x N1
    r12 = p12 @ s2                                # N1 x K2
    k0, k1, k2 = s0.shape[1], s1.shape[1], s2.shape[1]
    out = np.zeros((k0, k1, k2), dtype=np.float64)
    # Fast path for hard / nearly hard assignments.
    hardish = np.all((s1.max(axis=1) > 1 - 1e-10))
    if hardish:
        labels = np.argmax(s1, axis=1)
        for b in range(k1):
            mask = labels == b
            if np.any(mask):
                out[:, b, :] = w[:, mask] @ r12[mask, :]
    else:
        for b in range(k1):
            weights = s1[:, b]
            active = weights > 1e-10
            if np.any(active):
                out[:, b, :] = (w[:, active] * weights[active][None, :]) @ r12[active, :]
    out /= max(float(out.sum()), EPS)
    return out


def _optimal_time_file(
    closure_root: Path,
    time: str,
    time_points: tuple[str, ...],
    source_name: str,
    target_name: str,
) -> Path:
    root = Path(closure_root) / "optimal_coarse"
    try:
        index = time_points.index(time)
    except ValueError as exc:
        raise KeyError(time) from exc
    if index < len(time_points) - 1:
        pair = f"{time}_to_{time_points[index + 1]}"
        return root / pair / source_name
    pair = f"{time_points[index - 1]}_to_{time}"
    return root / pair / target_name


def _spot_order_from_optimal(
    closure_root: Path,
    time: str,
    time_points: tuple[str, ...] = TIMES,
) -> list[str]:
    path = _optimal_time_file(
        closure_root,
        time,
        time_points,
        "assignments_t.csv",
        "assignments_tp.csv",
    )
    return pd.read_csv(path)["spot_id"].astype(str).tolist()


def natural_assignment(
    data_root: Path,
    closure_root: Path,
    layer: str,
    time: str,
    organ: str = "heart",
    time_points: tuple[str, ...] = TIMES,
) -> np.ndarray:
    paths = layer_paths(data_root, organ, layer, time)
    state_units = read_index(paths["index"])
    spot_units = _spot_order_from_optimal(closure_root, time, time_points)
    return load_domain_assignment(paths["map"], spot_units, state_units)


def optimized_global_assignment(
    closure_root: Path,
    time: str,
    time_points: tuple[str, ...] = TIMES,
) -> np.ndarray:
    path = _optimal_time_file(closure_root, time, time_points, "S_t.npy", "S_tp.npy")
    return normalize_assignment(np.load(path))


def load_assignments(
    data_root: Path,
    closure_root: Path,
    organ: str = "heart",
    time_points: tuple[str, ...] = TIMES,
) -> dict[tuple[str, str], np.ndarray]:
    out: dict[tuple[str,str], np.ndarray] = {}
    for time in time_points:
        out[("Seurat K150", time)] = natural_assignment(
            data_root, closure_root, "seurat_k150", time, organ, time_points
        )
        out[("Seurat K40", time)] = natural_assignment(
            data_root, closure_root, "seurat_k40", time, organ, time_points
        )
        out[("Optimized coarse-graining", time)] = optimized_global_assignment(
            closure_root, time, time_points
        )
    return out


def load_spot_pijs(
    closure_root: Path,
    time_points: tuple[str, ...] = TIMES,
) -> dict[tuple[str,str], np.ndarray]:
    out = {}
    for a,b in zip(time_points[:-1], time_points[1:]):
        out[(a,b)] = row_normalize(load_matrix(Path(closure_root)/"matrices"/f"pij_spot_{a}_to_{b}.npz"))
    return out


def memory_order_tables(
    data_root: Path,
    closure_root: Path,
    lump_states: pd.DataFrame | None = None,
    organ: str = "heart",
    time_points: tuple[str, ...] = TIMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = load_assignments(data_root, closure_root, organ, time_points)
    pijs = load_spot_pijs(closure_root, time_points)
    summary_rows=[]; state_rows=[]
    triples = tuple(zip(time_points[:-2], time_points[1:-1], time_points[2:]))
    for t0,t1,t2 in triples:
        p01,p12=pijs[(t0,t1)],pijs[(t1,t2)]
        for mapping in MAPPINGS:
            joint=macro_triplet_joint(p01,p12,assignments[(mapping,t0)],assignments[(mapping,t1)],assignments[(mapping,t2)])
            cmi, per_b, pb=conditional_mi_triplet(joint)
            h1=conditional_entropy_c_given_b(joint); h2=conditional_entropy_c_given_ab(joint)
            p_ac=joint.sum(axis=1); long_mi=mi_joint(p_ac)
            summary_rows.append({
                "mapping":mapping,"window":f"{t0}->{t1}->{t2}","middle_time":t1,
                "markov_memory_cmi_bits":cmi,"H_future_given_current":h1,"H_future_given_current_and_past":h2,
                "history_prediction_gain_bits":h1-h2,
                "history_gain_fraction_of_uncertainty":cmi/max(h1,EPS),
                "past_future_mi_bits":long_mi,"memory_to_long_mi_ratio":cmi/max(long_mi,EPS),
                "active_middle_states":int(np.sum(pb>EPS)),
            })
            # Attach the existing next-step lumpability metric for the same middle state where possible.
            next_pair=f"{t1}->{t2}"
            lump=None
            if lump_states is not None:
                lump=lump_states[(lump_states["mapping"]==mapping)&(lump_states["time_pair"]==next_pair)].set_index("state_index")
            for b,(m,pbval) in enumerate(zip(per_b,pb)):
                row={"mapping":mapping,"window":f"{t0}->{t1}->{t2}","middle_time":t1,"state_index":b,
                     "state_memory_bits":float(m),"state_probability":float(pbval),"weighted_memory_contribution":float(m*pbval)}
                if lump is not None and b in lump.index:
                    r=lump.loc[b]
                    row.update({"next_step_lumpability_mean_js":float(r["mean_js"]),"next_step_q_best_direct_js":float(r["q_best_direct_js"]),"state_mass":float(r["mass"])})
                state_rows.append(row)
    return pd.DataFrame(summary_rows), pd.DataFrame(state_rows)


def macro_entropy(assignment: np.ndarray) -> tuple[float,float]:
    usage=normalize_assignment(assignment).mean(axis=0)
    usage=usage/usage.sum()
    h=entropy_prob(usage)
    return h,float(2**h)


def predictive_compression_table(
    data_root: Path,
    closure_root: Path,
    organ: str = "heart",
    time_points: tuple[str, ...] = TIMES,
) -> pd.DataFrame:
    summary=pd.read_csv(Path(closure_root)/"tables"/"closure_summary_all_scales.csv")
    assignments=load_assignments(data_root,closure_root,organ,time_points)
    rows=[]
    pairs = tuple(f"{a}->{b}" for a, b in zip(time_points[:-1], time_points[1:]))
    for pair in pairs:
        t0=pair.split("->")[0]
        spot=summary[(summary["mapping"]=="Spot (uncompressed)")&(summary["time_pair"]==pair)].iloc[0]
        n=int(spot["source_nodes"]); spot_h=np.log2(n)
        rows.append({"mapping":"Spot (uncompressed)","time_pair":pair,"macro_entropy_bits":spot_h,"effective_states":float(n),
                     "compression_fraction":1.0,"predictive_information_bits":float(spot["I_macro_to_future_macro"]),
                     "available_information_bits":float(spot["I_micro_to_future_macro"]),"macro_sufficiency":1.0,
                     "predictive_efficiency_bits_per_state_bit":float(spot["I_macro_to_future_macro"])/max(spot_h,EPS),
                     "closure_floor_js":0.0,"independent_q_excess_js":0.0})
        for mapping in MAPPINGS:
            s=assignments[(mapping,t0)]; h,keff=macro_entropy(s)
            r=summary[(summary["mapping"]==mapping)&(summary["time_pair"]==pair)].iloc[0]
            rows.append({"mapping":mapping,"time_pair":pair,"macro_entropy_bits":h,"effective_states":keff,
                         "compression_fraction":h/max(spot_h,EPS),"predictive_information_bits":float(r["I_macro_to_future_macro"]),
                         "available_information_bits":float(r["I_micro_to_future_macro"]),"macro_sufficiency":float(r["macro_sufficiency"]),
                         "predictive_efficiency_bits_per_state_bit":float(r["I_macro_to_future_macro"])/max(h,EPS),
                         "closure_floor_js":float(r["best_closure_mean_js"]),"independent_q_excess_js":float(r["direct_excess_js_above_best"])})
    return pd.DataFrame(rows)


def failure_taxonomy_table(lump_states: pd.DataFrame, quantile: float=0.75) -> tuple[pd.DataFrame,pd.DataFrame]:
    rows=[]; summary=[]
    for (mapping,pair),g in lump_states.groupby(["mapping","time_pair"],sort=False):
        t_intr=float(g["mean_js"].quantile(quantile)); t_q=float(g["q_best_direct_js"].quantile(quantile))
        gg=g.copy(); hi_intr=gg["mean_js"]>t_intr; hi_q=gg["q_best_direct_js"]>t_q
        cats=np.where(hi_intr & hi_q,"Dual-limited",np.where(hi_intr,"Partition-limited",np.where(hi_q,"Q-limited","Typical/closed")))
        gg["failure_class"]=cats; gg["intrinsic_threshold"]=t_intr; gg["q_threshold"]=t_q
        rows.append(gg)
        counts=gg["failure_class"].value_counts()
        for cls in ("Typical/closed","Partition-limited","Q-limited","Dual-limited"):
            sub=gg[gg["failure_class"]==cls]
            summary.append({"mapping":mapping,"time_pair":pair,"failure_class":cls,"states":int(len(sub)),"fraction":float(len(sub)/max(len(gg),1)),
                            "residual_kl_share":float(sub["residual_kl_share"].sum()),"mean_state_mass":float(sub["mass"].mean()) if len(sub) else np.nan})
    return pd.concat(rows,ignore_index=True),pd.DataFrame(summary)


def gini(values: np.ndarray) -> float:
    x=np.sort(np.maximum(np.asarray(values,float).reshape(-1),0.0))
    if len(x)==0 or x.sum()<=EPS:return 0.0
    n=len(x); return float((2*np.sum((np.arange(1,n+1))*x)/(n*x.sum()))-(n+1)/n)


def source_usage_from_assignment(s: np.ndarray, expected_states: int | None = None) -> np.ndarray:
    """Return usage aligned to the primary hard/active macro representation.

    Optimized assignments may have nominal empty prototype columns; the audited
    primary closure compacts those columns after argmax hardening.  When an
    expected state count is supplied, mirror that compaction here.
    """
    s=normalize_assignment(s)
    if expected_states is not None:
        labels=np.argmax(s,axis=1)
        active=np.unique(labels)
        if len(active)==expected_states:
            counts=np.asarray([(labels==a).sum() for a in active],dtype=float)
            return counts/max(counts.sum(),EPS)
    u=s.mean(axis=0)
    if expected_states is not None and len(u)!=expected_states:
        # Last-resort alignment for legacy cached arrays: select non-empty columns.
        nz=np.flatnonzero(u>EPS)
        if len(nz)==expected_states:
            u=u[nz]
        else:
            raise ValueError(f"Assignment/Q state mismatch: usage={len(u)}, expected={expected_states}, active={len(nz)}")
    return u/max(u.sum(),EPS)


def q_pair_paths(closure_root: Path, mapping: str, pair: str) -> tuple[np.ndarray,np.ndarray]:
    a,b=pair.split("->"); t=Path(closure_root)/"tables"
    if mapping=="Seurat K150": slug=f"seurat_k150_{a}_to_{b}"
    elif mapping=="Seurat K40": slug=f"seurat_k40_{a}_to_{b}"
    elif mapping=="Optimized coarse-graining": slug=f"optimized_{a}_to_{b}"
    else: raise KeyError(mapping)
    return row_normalize(np.load(t/f"q_best_{slug}.npy")),row_normalize(np.load(t/f"q_direct_{slug}.npy"))


def transition_mismatch_tables(
    data_root: Path,
    closure_root: Path,
    organ: str = "heart",
    time_points: tuple[str, ...] = TIMES,
) -> tuple[pd.DataFrame,pd.DataFrame]:
    assignments=load_assignments(data_root,closure_root,organ,time_points)
    summaries=[]; edges=[]
    pairs = tuple(f"{a}->{b}" for a, b in zip(time_points[:-1], time_points[1:]))
    for mapping in MAPPINGS:
        for pair in pairs:
            t0=pair.split("->")[0]; qb,qd=q_pair_paths(closure_root,mapping,pair); mu=source_usage_from_assignment(assignments[(mapping,t0)], expected_states=qb.shape[0])
            # q arrays can include nominal empty optimized states; source usage has same width K.
            signed=mu[:,None]*(qd-qb); absolute=np.abs(signed); flat=absolute.ravel(); total=float(flat.sum())
            order=np.argsort(flat)[::-1]
            top10=float(flat[order[:10]].sum()/max(total,EPS)); top25=float(flat[order[:25]].sum()/max(total,EPS)); top100=float(flat[order[:100]].sum()/max(total,EPS))
            summaries.append({"mapping":mapping,"time_pair":pair,"weighted_l1_mismatch":total,"top10_edge_share":top10,"top25_edge_share":top25,"top100_edge_share":top100,
                              "edge_mismatch_gini":gini(flat),"source_mismatch_gini":gini(absolute.sum(axis=1)),"target_mismatch_gini":gini(absolute.sum(axis=0)),
                              "mean_row_js":float(np.sum(mu*js_rows(qb,qd)))})
            for rank,idx in enumerate(order[:30],start=1):
                i,j=np.unravel_index(int(idx),absolute.shape)
                edges.append({"mapping":mapping,"time_pair":pair,"rank":rank,"source_state":i,"target_state":j,"signed_weighted_difference":float(signed[i,j]),
                              "absolute_weighted_difference":float(absolute[i,j]),"share_of_total_abs_mismatch":float(absolute[i,j]/max(total,EPS)),
                              "q_induced":float(qb[i,j]),"q_independent":float(qd[i,j]),"source_usage":float(mu[i])})
    return pd.DataFrame(summaries),pd.DataFrame(edges)


def subspace_alignment_table(
    closure_root: Path,
    ranks=(1,2,3,5,10),
    time_points: tuple[str, ...] = TIMES,
) -> pd.DataFrame:
    rows=[]
    pairs = tuple(f"{a}->{b}" for a, b in zip(time_points[:-1], time_points[1:]))
    for mapping in MAPPINGS:
        for pair in pairs:
            qb,qd=q_pair_paths(closure_root,mapping,pair)
            ub,sb,vhb=np.linalg.svd(qb,full_matrices=False); ud,sd,vhd=np.linalg.svd(qd,full_matrices=False)
            for r in ranks:
                rr=min(r,ub.shape[1],ud.shape[1])
                angles_l=scipy.linalg.subspace_angles(ub[:,:rr],ud[:,:rr]); angles_r=scipy.linalg.subspace_angles(vhb[:rr,:].T,vhd[:rr,:].T)
                rows.append({"mapping":mapping,"time_pair":pair,"rank":r,
                             "left_subspace_cosine_mean":float(np.mean(np.cos(angles_l))),"left_subspace_cosine_min":float(np.min(np.cos(angles_l))),
                             "right_subspace_cosine_mean":float(np.mean(np.cos(angles_r))),"right_subspace_cosine_min":float(np.min(np.cos(angles_r))),
                             "dominant_left_vector_alignment":float(abs(np.dot(ub[:,0],ud[:,0]))),"dominant_right_vector_alignment":float(abs(np.dot(vhb[0,:],vhd[0,:]))),
                             "singular_value_cosine":float(np.dot(sb,sd)/(max(np.linalg.norm(sb)*np.linalg.norm(sd),EPS)))})
    return pd.DataFrame(rows)


def cross_scale_information_anatomy(
    closure_root: Path,
    time_points: tuple[str, ...] = TIMES,
) -> pd.DataFrame:
    summary=pd.read_csv(Path(closure_root)/"tables"/"closure_summary_all_scales.csv")
    rows=[]
    pairs = tuple(f"{a}->{b}" for a, b in zip(time_points[:-1], time_points[1:]))
    for mapping in MAPPINGS:
        for pair in pairs:
            r=summary[(summary["mapping"]==mapping)&(summary["time_pair"]==pair)].iloc[0]
            rows.append({"mapping":mapping,"time_pair":pair,
                         "upward_macro_sufficiency":float(r["macro_sufficiency"]),
                         "upward_residual_fraction":float(r["residual_fraction"]),
                         "downward_future_micro_reach":float(r["downward_reach_ratio"]),
                         "macro_future_macro_bits":float(r["I_macro_to_future_macro"]),
                         "macro_future_micro_bits":float(r["I_macro_to_future_micro"]),
                         "micro_future_macro_bits":float(r["I_micro_to_future_macro"]),
                         "micro_future_micro_bits":float(r["EI_micro_dynamics"]),
                         "closure_floor_js":float(r["best_closure_mean_js"]),
                         "independent_q_excess_js":float(r["direct_excess_js_above_best"]),
                         "closure_autonomy_score":float(r["macro_sufficiency"])*(1-float(r["best_closure_mean_js"]))})
    return pd.DataFrame(rows)


def state_memory_correlations(memory_states: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (mapping,window),g in memory_states.groupby(["mapping","window"],sort=False):
        for x in ("next_step_lumpability_mean_js","next_step_q_best_direct_js","state_mass"):
            if x not in g.columns: continue
            sub=g[["state_memory_bits",x,"state_probability"]].dropna()
            if len(sub)<3: continue
            pear=float(sub["state_memory_bits"].corr(sub[x],method="pearson")); spear=float(sub["state_memory_bits"].corr(sub[x],method="spearman"))
            rows.append({"mapping":mapping,"window":window,"x":x,"pearson_r":pear,"spearman_rho":spear,"n":len(sub)})
    return pd.DataFrame(rows)


def _hard_closure_floor(labels: np.ndarray, future_macro: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Uniform-source intrinsic closure floor for a hard source partition.

    Returns global mean JS, per-source JS, and one centroid row per hard state id.
    """
    labels=np.asarray(labels,dtype=int)
    r=row_normalize(future_macro)
    k=int(labels.max())+1 if labels.size else 0
    q=np.zeros((k,r.shape[1]),dtype=float)
    for s in range(k):
        idx=np.flatnonzero(labels==s)
        if idx.size:
            q[s]=r[idx].mean(axis=0)
        else:
            q[s]=1.0/max(r.shape[1],1)
    q=row_normalize(q)
    pred=q[labels]
    js=js_rows(r,pred)
    return float(js.mean()),js,q


def _binary_future_split(indices: np.ndarray, future_macro: np.ndarray, max_iter: int=40) -> np.ndarray | None:
    """Deterministic two-means split in square-root probability geometry.

    This is a downstream diagnostic, not a model re-fit: it asks whether the
    worst macrostate secretly contains two kinetically distinct future profiles.
    """
    idx=np.asarray(indices,dtype=int)
    if idx.size < 4:
        return None
    x=np.sqrt(row_normalize(future_macro[idx]))
    mean=x.mean(axis=0)
    a=int(np.argmax(np.sum((x-mean)**2,axis=1)))
    b=int(np.argmax(np.sum((x-x[a])**2,axis=1)))
    if a==b:
        return None
    centers=np.stack([x[a],x[b]],axis=0)
    lab=np.zeros(idx.size,dtype=int)
    for _ in range(max_iter):
        d=((x[:,None,:]-centers[None,:,:])**2).sum(axis=2)
        new=np.argmin(d,axis=1)
        if np.all(new==lab) and np.all(np.bincount(new,minlength=2)>0):
            break
        lab=new
        if np.any(np.bincount(lab,minlength=2)==0):
            return None
        centers=np.stack([x[lab==c].mean(axis=0) for c in (0,1)],axis=0)
    if np.any(np.bincount(lab,minlength=2)<2):
        return None
    return lab


def closure_repairability_tables(
    data_root: Path,
    closure_root: Path,
    organ: str = "heart",
    time_points: tuple[str, ...] = TIMES,
    max_splits: int = 8,
    random_repeats: int = 40,
    seed: int = 20260809,
) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Counterfactual state-refinement diagnostic for approximate closure.

    Iteratively split the source macrostate carrying the largest intrinsic
    closure residual, using only its future-macro probability profiles.  The
    analysis quantifies how many *targeted* state refinements are required to
    remove a given fraction of the closure floor, and compares each split with
    random within-state splitting of the same state.

    This deliberately does not change the trained model and is therefore a
    downstream *repairability/headroom* diagnostic rather than optimization.
    """
    assignments=load_assignments(data_root,closure_root,organ,time_points)
    pijs=load_spot_pijs(closure_root,time_points)
    rng=np.random.default_rng(seed)
    curve_rows=[];split_rows=[]
    pairs = tuple(f"{a}->{b}" for a, b in zip(time_points[:-1], time_points[1:]))
    for pair in pairs:
        t0,t1=pair.split('->'); p=pijs[(t0,t1)]
        for mapping in MAPPINGS:
            s0=assignments[(mapping,t0)]; s1=assignments[(mapping,t1)]
            future=row_normalize(p @ s1)
            labels=np.argmax(s0,axis=1).astype(int)
            # Relabel compactly because optimized hard assignments can leave empty nominal states.
            _,labels=np.unique(labels,return_inverse=True)
            base,js,_=_hard_closure_floor(labels,future)
            curve_rows.append({'mapping':mapping,'time_pair':pair,'added_states':0,'hard_state_count':int(np.unique(labels).size),'closure_floor_js':base,'fraction_floor_remaining':1.0,'fraction_floor_removed':0.0})
            current=base
            for step in range(1,max_splits+1):
                # Choose the state carrying the largest total residual mass and that can be split.
                candidates=[]
                for st in np.unique(labels):
                    idx=np.flatnonzero(labels==st)
                    if idx.size>=4:
                        candidates.append((float(js[idx].sum()),int(st),idx))
                if not candidates:
                    break
                candidates.sort(reverse=True,key=lambda z:z[0])
                chosen=None
                for _,st,idx in candidates:
                    sub=_binary_future_split(idx,future)
                    if sub is not None:
                        chosen=(st,idx,sub);break
                if chosen is None:
                    break
                st,idx,sub=chosen
                new_label=int(labels.max())+1
                targeted=labels.copy(); targeted[idx[sub==1]]=new_label
                new_floor,new_js,_=_hard_closure_floor(targeted,future)
                target_gain=current-new_floor
                # Matched random split of exactly the same state and group sizes.
                n1=int(np.sum(sub==1));rand_gains=[]
                for _ in range(random_repeats):
                    perm=rng.permutation(idx)
                    rr=labels.copy();rr[perm[:n1]]=new_label
                    rf,_,_=_hard_closure_floor(rr,future);rand_gains.append(current-rf)
                rand_gains=np.asarray(rand_gains,float)
                mass=int(idx.size)
                split_rows.append({'mapping':mapping,'time_pair':pair,'split_step':step,'source_state':int(st),'state_mass':mass,'closure_before_js':current,'closure_after_js':new_floor,'targeted_gain_js':target_gain,'random_gain_mean_js':float(np.mean(rand_gains)),'random_gain_sd_js':float(np.std(rand_gains)),'gain_over_random_ratio':float(target_gain/max(float(np.mean(rand_gains)),EPS)) if np.mean(rand_gains)>EPS else np.nan,'child1_mass':int(np.sum(sub==0)),'child2_mass':int(np.sum(sub==1))})
                labels=targeted;current=new_floor;js=new_js
                curve_rows.append({'mapping':mapping,'time_pair':pair,'added_states':step,'hard_state_count':int(np.unique(labels).size),'closure_floor_js':current,'fraction_floor_remaining':current/max(base,EPS),'fraction_floor_removed':1-current/max(base,EPS)})
    return pd.DataFrame(curve_rows),pd.DataFrame(split_rows)

def run_ultradeep_closure_analysis(
    *,
    data_root: Path,
    closure_root: Path,
    deep_root: Path,
    output_root: Path,
    organ: str = "heart",
    time_points: tuple[str, ...] = TIMES,
    max_splits: int = 8,
    random_repeats: int = 40,
    seed: int = 20260809,
) -> dict[str, Any]:
    data_root = Path(data_root)
    closure_root = Path(closure_root)
    deep_root = Path(deep_root)
    output_root = Path(output_root)
    tables = output_root / "tables"
    figures = output_root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    lump_states = pd.read_csv(deep_root / "tables" / "closure_lumpability_states.csv")
    memory_summary, memory_states = memory_order_tables(
        data_root, closure_root, lump_states, organ, time_points
    )
    memory_corr = state_memory_correlations(memory_states)
    predictive = predictive_compression_table(
        data_root, closure_root, organ, time_points
    )
    failure_states, failure_summary = failure_taxonomy_table(lump_states)
    mismatch_summary, mismatch_edges = transition_mismatch_tables(
        data_root, closure_root, organ, time_points
    )
    subspace = subspace_alignment_table(closure_root, time_points=time_points)
    info_anatomy = cross_scale_information_anatomy(closure_root, time_points)
    repair_curve, repair_splits = closure_repairability_tables(
        data_root,
        closure_root,
        organ,
        time_points,
        max_splits,
        random_repeats,
        seed,
    )
    frames = {
        "closure_markov_memory_summary.csv": memory_summary,
        "closure_markov_memory_states.csv": memory_states,
        "closure_markov_memory_correlations.csv": memory_corr,
        "closure_predictive_compression.csv": predictive,
        "closure_failure_taxonomy_states.csv": failure_states,
        "closure_failure_taxonomy_summary.csv": failure_summary,
        "closure_transition_mismatch_summary.csv": mismatch_summary,
        "closure_transition_mismatch_top_edges.csv": mismatch_edges,
        "closure_dynamic_subspace_alignment.csv": subspace,
        "closure_cross_scale_information_anatomy.csv": info_anatomy,
        "closure_repairability_curve.csv": repair_curve,
        "closure_repairability_splits.csv": repair_splits,
    }
    for name, frame in frames.items():
        frame.to_csv(tables / name, index=False)
    paths = render_ultradeep_figures(
        memory_summary=memory_summary,
        memory_states=memory_states,
        memory_corr=memory_corr,
        predictive=predictive,
        failure_states=failure_states,
        failure_summary=failure_summary,
        mismatch_summary=mismatch_summary,
        mismatch_edges=mismatch_edges,
        subspace=subspace,
        info_anatomy=info_anatomy,
        repair_curve=repair_curve,
        repair_splits=repair_splits,
        output_dir=figures,
    )
    findings = {
        "scope": "Ultra-deep dynamical closure: explicit macro Markov-memory CMI, predictive compression frontier, state-level failure taxonomy, transition mismatch anatomy, singular-subspace alignment, cross-scale information anatomy, and targeted state-refinement repairability.",
        "figures": [str(path) for path in paths],
        "tables": list(frames),
    }
    (output_root / "findings.json").write_text(
        json.dumps(findings, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    manifest = write_manifest(
        output_root,
        {
            "data_root": str(data_root.resolve()),
            "closure_root": str(closure_root.resolve()),
            "deep_root": str(deep_root.resolve()),
            "organ": organ,
            "time_points": list(time_points),
            "max_splits": max_splits,
            "random_repeats": random_repeats,
            "seed": seed,
        },
    )
    return {
        "output_root": output_root,
        "figures": paths,
        "tables": frames,
        "findings": findings,
        "manifest": manifest,
    }
