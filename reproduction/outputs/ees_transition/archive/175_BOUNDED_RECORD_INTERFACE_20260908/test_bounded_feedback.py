"""Exact-grid, random, fail-closed, and zero-radius interface tests."""
from pathlib import Path
from fractions import Fraction
import hashlib,itertools,json,inspect
import numpy as np
from bounded_feedback import assess_bounded_record as assess
A=Path(__file__).resolve().parent
def truth(x,y,xr,a,tol):
    idle=np.asarray(x)-y>tol;o=~idle;yr=np.minimum(xr,a)
    return bool(idle.any() and np.maximum(np.asarray(xr)[o]-np.asarray(x)[o],0).sum()>tol
                and np.maximum(yr[o]-np.asarray(y)[o],0).sum()>tol and (yr-y).sum()>tol)
def run():
    n=0
    # Powers are dyadic rationals, so their binary floating representations are exact.
    vals=[Fraction(i,4) for i in range(5)]
    for a,x,xr,d in itertools.product(vals,repeat=4):
        y=min(x,a)
        for e in [-d,Fraction(0),d]:
            h=max(y,a+e);low=max(y,h-d)
            assert low<=a and min(xr,low)<=min(xr,a)
            n+=1
    rng=np.random.default_rng(175);passing=0
    for k in range(20000):
        size=int(rng.integers(2,45));cap=rng.uniform(.001,.1,size)
        a=rng.uniform(0,1.5,size)*cap;x=rng.uniform(0,1,size)*cap
        y=np.minimum(x,a);xr=rng.uniform(0,1,size)*cap
        delta=rng.uniform(0,.2,size)*cap;h=np.maximum(y,a+rng.uniform(-1,1,size)*delta)
        base=dict(allocation=x,delivery=y,replay_allocation=xr,record_valid=True,replay_physical_valid=True,tolerance=2e-6)
        q=assess(**base,record=h,error_radius=delta)
        assert q.record_valid
        assert np.max(np.asarray(q.lower_availability)-a)<1e-14
        assert np.max(np.asarray(q.lower_replay_delivery)-np.minimum(xr,a))<1e-14
        if q.qualifies:assert truth(x,y,xr,a,2e-6);passing+=1
        larger=assess(**base,record=h,error_radius=delta*2)
        assert not larger.qualifies or q.qualifies
        exact=assess(**base,record=a,error_radius=np.zeros(size))
        assert exact.qualifies==truth(x,y,xr,a,2e-6)
        invalid=assess(**{**base,'replay_physical_valid':False},record=h,error_radius=delta)
        assert not invalid.qualifies
    good=dict(allocation=[2.,1.],delivery=[1.,1.],replay_allocation=[1.,2.],record=[1.,2.],error_radius=[0.,0.],record_valid=True,replay_physical_valid=True,tolerance=.01)
    assert assess(**good).qualifies
    faults=[{'record':None},{'error_radius':None},{'record_valid':False},{'record_valid':1},
            {'replay_physical_valid':1},{'record':[1.]},{'error_radius':[-1.,0.]},
            {'record':[float('nan'),2.]},{'record':[float('inf'),2.]},
            {'delivery':[3.,1.]},{'record':[0.,2.]},{'tolerance':-1.},
            {'tolerance':float('nan')},{'allocation':[]},{'record':['bad',2.]}]
    for f in faults:assert not assess(**{**good,**f}).record_valid,f
    # Same disclosed values lead to identical decisions regardless of imagined truths.
    assert 'availability' not in inspect.signature(assess).parameters
    try:assess(**good,true_availability=[1.,2.])
    except TypeError:pass
    else:raise AssertionError('Latent input unexpectedly accepted')
    out={'exact_rational_lower_bound_cases':n,'random_vector_cases':20000,
         'random_qualifying_cases_checked_against_truth':passing,'invalid_input_cases':len(faults),
         'zero_radius_and_monotonicity_checks':20000,'latent_argument_rejected':True,
         'interface_sha256':hashlib.sha256((A/'bounded_feedback.py').read_bytes()).hexdigest()}
    (A/'UNIT_TESTS.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(out)
if __name__=='__main__':run()
