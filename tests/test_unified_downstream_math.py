import numpy as np
from mignet_ce.visualization.downstream.unified_suite import closure_budget, crossfit_error, hard_compact

def H(labels,k=None):
    labels=np.asarray(labels,int); k=(labels.max()+1) if k is None else k; out=np.zeros((len(labels),k)); out[np.arange(len(labels)),labels]=1; return out

def test_unified_downstream_math():
    # Exact cross-time lumpability: microstates within a source macro share future-macro profiles.
    p=np.array([[.7,.3,0,0],[.2,.8,0,0],[0,0,.6,.4],[0,0,.1,.9]])
    hs=H([0,0,1,1]); ht=H([0,0,1,1]); b=closure_budget(p,hs,ht)
    assert b['closure_leakage_bits'] < 1e-10
    assert b['identity_error'] < 1e-10
    # Non-lumpable: same source macro contains opposite macro futures.
    p2=np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]],float)
    b2=closure_budget(p2,hs,ht)
    assert b2['closure_leakage_bits'] > 0.5
    # Low signal: identical macro future profile for every microstate => ratio must be undefined/NA.
    p3=np.tile(np.array([[.5,.5,0,0]]),(4,1)); b3=closure_budget(p3,hs,ht)
    assert b3['signal_status']=='low-signal' and np.isnan(b3['closure_quality'])
    # Soft vectors with the same hard labels must not alter the primary closure result.
    s1=np.array([[.9,.1],[.51,.49],[.1,.9],[.49,.51]])
    s2=np.array([[.7,.3],[.99,.01],[.4,.6],[.01,.99]])
    q1=closure_budget(p2,s1,ht)['closure_leakage_bits']; q2=closure_budget(p2,s2,ht)['closure_leakage_bits']
    assert abs(q1-q2)<1e-12
    # Singleton cross-fit is excluded, not self-predicted with zero error.
    obs=np.array([[1,0],[.8,.2],[.2,.8],[0,1]],float); h=H([0,1,1,1]); w=np.full(4,.25)
    cf=crossfit_error(obs,h,w,folds=3,seed=1)
    assert cf['crossfit_coverage'] < 1.0 and cf['singleton_weight'] > 0
    # Active-state compaction must use actual occupied states, not nominal K.
    soft=np.zeros((5,10)); soft[np.arange(5),[0,0,3,3,3]]=1; hh,active=hard_compact(soft)
    assert hh.shape[1]==2 and len(active)==2
    print('7/7 unified downstream math checks passed')
if __name__=='__main__': test_unified_downstream_math()
