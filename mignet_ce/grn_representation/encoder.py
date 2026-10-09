"""Pair-scoped structural GRN enhancer with NumPy/Torch parity.

- Align source and target matrices to the union of their *pair* gene names.
- Fit structural basis on source edges only (target only sets dimensions).
- Cache one reference scale from source spot features; do not refit for macro.
- Preserve the historical per-stage gene-hash Legacy projection verbatim.

The pair-union policy deliberately differs from the former Round-6
all-stage union; consequently numerical parity with Round-6 is not promised.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import torch
from dataclasses import dataclass
from .config import GRNFeatureConfig
from .basis import fit_nmf_basis,fit_lap_hks_basis

def _union_adjacency(w:sp.spmatrix, old_genes: list[str], gene_map: dict[str,int], n:int)->sp.csr_matrix:
    a=w.tocoo()
    indexes=np.asarray([gene_map[g] for g in old_genes],dtype=np.int64)
    return sp.coo_matrix((a.data,(indexes[a.row],indexes[a.col])),shape=(n,n)).tocsr()

def _proj(width: int,output: int,seed:int)->np.ndarray:
    rng=np.random.default_rng(seed)
    return rng.normal(0,1/np.sqrt(width),(width,output)).astype(np.float32)

def _rms(x:np.ndarray)->float:
    return float(np.sqrt(np.mean(np.asarray(x,dtype=np.float64)**2)))

@dataclass
class GRNPairEnhancer:
    config:GRNFeatureConfig
    gene_count:int
    source_basis:tuple[np.ndarray,np.ndarray] | np.ndarray
    projection:np.ndarray
    basis_t:tuple[np.ndarray,np.ndarray] | np.ndarray
    basis_tp:tuple[np.ndarray,np.ndarray] | np.ndarray
    scale:float
    metadata:dict

    @classmethod
    def fit(cls, stage_t,stage_tp, config:GRNFeatureConfig):
        config.validate()
        if config.method=='legacy':
            raise ValueError('Legacy does not require a GRNPairEnhancer.')
        genes=sorted(set(stage_t.grn_genes)|set(stage_tp.grn_genes))
        lookup={g:i for i,g in enumerate(genes)}
        w=_union_adjacency(stage_t.grn_adjacency,stage_t.grn_genes,lookup,len(genes))
        if config.method=='legacy_nmf':
            basis=fit_nmf_basis(w,rank=config.nmf_rank,steps=config.nmf_steps,seed=config.nmf_seed)
            project=_proj(config.nmf_rank*2,stage_t.g_raw.shape[1],20261009)
        else:
            basis=fit_lap_hks_basis(w,rank=config.hks_rank)
            project=_proj(basis.shape[1]*2,stage_t.g_raw.shape[1],20261010)
        def subset(stage):
            ids=np.asarray([lookup[g] for g in stage.grn_genes],dtype=int)
            return (basis[0][ids],basis[1][ids]) if isinstance(basis,tuple) else basis[ids]
        t,tp=subset(stage_t),subset(stage_tp)
        cls0=cls(config,len(genes),basis,project,t,tp,1.,{})
        extra=cls0.residual_numpy(stage_t.expression_grn,stage_t.grn_adjacency,'t')
        scale=_rms(stage_t.g_raw)/max(_rms(extra),1e-9)
        cls0.scale=scale
        cls0.metadata={'method':config.method,'residual_lambda':config.residual_lambda,'source_only_basis':True,
            'gene_space':'source_target_pair_union','gene_count_union':len(genes),
            'source_genes':len(stage_t.grn_genes),'target_genes':len(stage_tp.grn_genes),
            'nmf_rank':config.nmf_rank if config.method=='legacy_nmf' else None,
            'nmf_steps':config.nmf_steps if config.method=='legacy_nmf' else None,
            'hks_times':[.5,1.,2.,4.,8.] if config.method=='legacy_lap_hks' else None,
            'residual_reference_scale':float(scale),'reference_rms':_rms(stage_t.g_raw),
            'basis_projection_output_dim':int(project.shape[1]),
            'projection_seed':20261009 if config.method=='legacy_nmf' else 20261010,
            'note':'Pair-union differs from Round-6 all-stage union; source basis fitted on source edges.'}
        return cls0

    def _factors(self,label:str):
        if label not in {'t','tp'}:raise ValueError('label must be t or tp')
        return self.basis_t if label=='t' else self.basis_tp

    def residual_numpy(self,expression:np.ndarray,adjacency:sp.spmatrix,label:str)->np.ndarray:
        x=np.maximum(np.nan_to_num(np.asarray(expression,dtype=np.float32)),0)
        a=adjacency.tocsr()
        r=x*np.asarray(a@x.T).T
        t=x*np.asarray(a.T@x.T).T
        b=self._factors(label)
        if isinstance(b,tuple):embedded=np.concatenate([r@b[0],t@b[1]],axis=1)
        else:embedded=np.concatenate([r@b,t@b],axis=1)
        return np.asarray(embedded@self.projection,dtype=np.float32)

    def feature_numpy(self,expression,stage,label):
        from wyt_deltaei_coarse_grain.complete_combined import project_grn_state
        base=project_grn_state(expression,stage.grn_adjacency,stage.projection_reg,stage.projection_tar)
        return base+self.config.residual_lambda*self.scale*self.residual_numpy(expression,stage.grn_adjacency,label)

    def residual_torch(self,expression:torch.Tensor,adjacency:torch.Tensor,label:str)->torch.Tensor:
        x=torch.clamp(torch.nan_to_num(expression,nan=0.,posinf=0.,neginf=0.),min=0.)
        r=x*torch.sparse.mm(adjacency,x.T).T
        t=x*torch.sparse.mm(adjacency.transpose(0,1),x.T).T
        b=self._factors(label)
        bcast=lambda a:torch.as_tensor(a,dtype=x.dtype,device=x.device)
        if isinstance(b,tuple):embedded=torch.cat([r@bcast(b[0]),t@bcast(b[1])],dim=1)
        else:embedded=torch.cat([r@bcast(b),t@bcast(b)],dim=1)
        return embedded@bcast(self.projection)

    def feature_torch(self,expression,adjacency,projection_reg,projection_tar,label:str):
        x=torch.clamp(torch.nan_to_num(expression,nan=0.,posinf=0.,neginf=0.),min=0.)
        r=x*torch.sparse.mm(adjacency,x.T).T
        t=x*torch.sparse.mm(adjacency.transpose(0,1),x.T).T
        base=r@projection_reg+t@projection_tar
        return base+self.config.residual_lambda*self.scale*self.residual_torch(x,adjacency,label)
