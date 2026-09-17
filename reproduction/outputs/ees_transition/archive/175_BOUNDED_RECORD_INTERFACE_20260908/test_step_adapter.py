from pathlib import Path
import hashlib,json,inspect
import numpy as np
from step_adapter import completed_feedback_step,old
A=Path(__file__).resolve().parent
rng=np.random.default_rng(1751);checks=0
for beta in [.25,.5,1.]:
    for trial in range(40):
        cap=rng.uniform(.01,.03,20);state=cap.copy()
        for t in range(96):
            raw=rng.uniform(.2,1.,20)*cap
            req=np.minimum(raw,state)
            network_budget=.5*cap.sum()
            x=req*min(1.,network_budget/req.sum())
            a=rng.uniform(0,1.,20)*cap;y=np.minimum(x,a)
            idle=x-y>2e-6;rplreq=np.where(idle,y,req)
            xr=rplreq*min(1.,network_budget/rplreq.sum())
            delta=.05*cap;record=np.maximum(y,a+rng.uniform(-1,1,20)*delta)
            valid=t%13!=0;physical=t%17!=0
            # No exact availability enters the step; the generator is outside it.
            q,after,target=completed_feedback_step(capacity=cap,ceiling_before=state,
                allocation=x,delivery=y,replay_allocation=xr,record=record,
                error_radius=delta if t%19 else None,record_valid=valid,
                replay_physical_valid=physical,tolerance=2e-6,beta=beta)
            expected=cap.copy()
            if q.qualifies:expected[idle]=y[idle]
            assert np.max(np.abs(expected-target))<1e-15
            assert np.max(np.abs(np.asarray(after)-((1-beta)*state+beta*expected)))<1e-15
            assert np.min(after)>=0 and np.max(np.asarray(after)-cap)<=1e-15
            if not valid or not physical or t%19==0:
                assert not q.qualifies and np.array_equal(target,cap)
            state=np.asarray(after);checks+=1
assert 'availability' not in inspect.signature(completed_feedback_step).parameters
out={'synthetic_recursive_steps':checks,'gains':[.25,.5,1.],
     'missing_or_invalid_record_follows_capacity_recovery':True,
     'false_physical_predicate_follows_capacity_recovery':True,
     'unmodified_original_update_function_used':True,
     'adapter_sha256':hashlib.sha256((A/'step_adapter.py').read_bytes()).hexdigest(),
     'scope':'Interface integration/unit test on a scalar shared-budget fixture, not a new feeder experiment.'}
(A/'STEP_TESTS.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(out)
