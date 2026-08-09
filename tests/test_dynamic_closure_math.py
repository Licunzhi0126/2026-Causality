import numpy as np
from mignet_ce.visualization.downstream.dynamic_closure.analysis import (
    information_closure_budget, induced_macro_q, to_hard_assignment,
    crossfit_macro_q_error, row_normalize,
)


def onehot(labels,k):
    x=np.zeros((len(labels),k));x[np.arange(len(labels)),labels]=1;return x


def test_exact_lumpable_identity_and_zero_leakage():
    # states 0,1 -> A and 2,3 -> B; members of each macro share identical macro future.
    P=np.array([
        [.5,.5,0,0], [.5,.5,0,0], [0,0,.5,.5], [0,0,.5,.5]
    ],float)
    H=onehot([0,0,1,1],2)
    b=information_closure_budget(P,H,H,weighting='state_balanced',low_signal_threshold_bits=0)
    assert b['closure_leakage_bits'] < 1e-10
    assert b['information_identity_error'] < 1e-10
    assert np.isclose(b['macro_sufficiency'],1.0)


def test_non_lumpable_has_positive_leakage():
    P=np.eye(4)
    H=onehot([0,0,1,1],2)
    # Future macro is deterministic by micro but current macro groups mixed future labels if target map is crossed.
    Ht=onehot([0,1,0,1],2)
    b=information_closure_budget(P,H,Ht,weighting='state_balanced',low_signal_threshold_bits=0)
    assert b['closure_leakage_bits'] > .9
    assert b['I_macro_retained_bits'] < 1e-10


def test_low_signal_marks_sufficiency_nan():
    P=np.full((4,4),.25)
    H=onehot([0,0,1,1],2)
    b=information_closure_budget(P,H,H,weighting='state_balanced')
    assert b['signal_status']=='low-signal'
    assert np.isnan(b['macro_sufficiency'])


def test_soft_hard_fairness():
    S=np.array([[.9,.1],[.51,.49],[.1,.9],[.49,.51]])
    H=to_hard_assignment(S)
    assert np.array_equal(np.argmax(H,1),np.array([0,0,1,1]))
    Q=np.array([[1,0],[0,1]],float)
    # old soft coordinate makes two same-hard-label spots predict differently
    old=row_normalize(S@Q)
    new=row_normalize(H@Q)
    assert not np.allclose(old[0],old[1])
    assert np.allclose(new[0],new[1])


def test_induced_q_is_kl_centroid_for_hard_group():
    obs=np.array([[.9,.1],[.7,.3],[.2,.8],[.4,.6]])
    H=onehot([0,0,1,1],2)
    q,m=induced_macro_q(obs,H,np.full(4,.25))
    assert np.allclose(q[0],[.8,.2])
    assert np.allclose(q[1],[.3,.7])
    assert np.allclose(m,[.5,.5])


def test_crossfit_exact_lumpable_zero():
    obs=np.array([[.8,.2],[.8,.2],[.1,.9],[.1,.9]])
    H=onehot([0,0,1,1],2)
    r=crossfit_macro_q_error(obs,H,folds=2)
    assert r['crossfit_kl_bits'] < 1e-10
    assert r['crossfit_js'] < 1e-10


def test_crossfit_excludes_singleton_states():
    obs=np.array([[1,0],[.8,.2],[.2,.8],[0,1]],float)
    H=onehot([0,1,1,1],2)
    r=crossfit_macro_q_error(obs,H,folds=3,seed=1)
    assert r['crossfit_coverage'] < 1.0
    assert r['singleton_weight_fraction'] > 0.0
