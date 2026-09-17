"""Fault injection into copies of completed vectors, never experiment files."""
from pathlib import Path
import contextlib,copy,json
from unittest.mock import patch
import numpy as np
import run_scale as run
import verify_scale as check

A=Path(__file__).resolve().parent

def main():
    p=run.verify_sources()
    out=A/'days'/p['settings'][0]['id']/p['dates'][0]
    done=run.completed(out);assert done
    baseline=json.loads((out/'RESULT.json').read_text('utf-8'))
    with np.load(out/'VECTORS.npz') as z:original={k:z[k].copy() for k in z.files}
    participants=np.flatnonzero(original['participant'])
    live=np.argwhere(original['award_mw']>1e-5)[0]
    call,person=map(int,live)
    # Choose a nonempty common-control request so preserving its total alone
    # cannot hide a wrong distribution among connections.
    t=int(np.flatnonzero(original['request_mw'].reshape(48,5,-1)[:,4].sum(axis=1)>1e-4)[0])
    u=5*t+4
    donor=int(np.flatnonzero(original['request_mw'][u]>1e-5)[0])
    receiver=int(next(i for i in participants if i!=donor))

    def evaluate(z,r):
        original_read=Path.read_text
        def read(path,*args,**kwargs):
            if path==out/'RESULT.json':return json.dumps(r)
            return original_read(path,*args,**kwargs)
        with patch.object(run,'completed',return_value=done),\
             patch.object(check.np,'load',return_value=contextlib.nullcontext(z)),\
             patch.object(Path,'read_text',read):
            return check.inspect_day(out)

    result,bad=evaluate(original,baseline)
    assert not bad
    def uniform_distribution(z,r):
        z['request_mw'][u,donor]-=1e-6;z['request_mw'][u,receiver]+=1e-6
    def wrong_daily_energy(z,r):r['day_summaries'][0]['delivered_export_mwh']+=.01
    def wrong_record_energy(z,r):r['interval_records'][0]['authorized_export_kw']+=1
    cases=[
        ('clipped_delivery',lambda z,r:z['delivery_mw'].__setitem__((call,person),z['delivery_mw'][call,person]+1e-5)),
        ('negative_award',lambda z,r:z['award_mw'].__setitem__((call,person),-1e-5)),
        ('qualification_flag',lambda z,r:z['gate'].__setitem__(t,not z['gate'][t])),
        ('ceiling_recursion',lambda z,r:z['ceiling_after'].__setitem__((t,person),z['ceiling_after'][t,person]+1e-4)),
        ('recovery_target',lambda z,r:z['ceiling_target'].__setitem__((t,person),z['ceiling_target'][t,person]+1e-4)),
        ('uniform_distribution_same_total',uniform_distribution),
        ('different_load_across_controls',lambda z,r:z['load_kw'].__setitem__((u,person),z['load_kw'][u,person]+1)),
        ('different_available_export',lambda z,r:z['available_mw'].__setitem__((u,person),z['available_mw'][u,person]+1e-4)),
        ('wrong_record_total',wrong_record_energy),
        ('wrong_daily_total',wrong_daily_energy),
    ]
    rejected=[]
    for name,mutate in cases:
        z={k:v.copy() for k,v in original.items()};r=copy.deepcopy(baseline)
        mutate(z,r)
        try:evaluate(z,r)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Verifier accepted corrupted output: '+name)
    assert run.completed(out)==done
    run.verify_sources()
    report=dict(passed=True,baseline_setting=p['settings'][0]['id'],baseline_date=p['dates'][0],
        rejected_faults=rejected,source_files_unchanged=True,experiment_files_unchanged=True,
        test_sha256=run.sha(Path(__file__)),protocol_sha256=run.sha(A/'PROTOCOL.json'),
        scope='In-memory failure injection into the saved-vector checker; not additional physical simulation or proof of all possible faults.')
    (A/'SAVED_CHECKER_TESTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
