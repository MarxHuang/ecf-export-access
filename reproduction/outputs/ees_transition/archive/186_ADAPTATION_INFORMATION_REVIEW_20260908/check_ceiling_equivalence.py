"""A request rewrite at the existing ceiling is not a new adaptation test."""
from pathlib import Path
import hashlib
import json
import numpy as np

A=Path(__file__).resolve().parent
S=A.parent/'169_SAME_UPDATE_QUALIFICATION_COMPARISON_20260908'
P=A.parents[1]/'EES_PARAGRAPH_REVIEW_MASTER_20260901'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def admitted(raw,capacity,ceiling):
    return np.minimum(raw,np.minimum(capacity,ceiling))


def main():
    protected={str(p):sha(p) for p in [P/'main.tex',P/'supporting_information.tex',
              P/'main.pdf',P/'supporting_information.pdf',*sorted((P/'figures').glob('*.pdf'))]}
    sources={}
    def read(p):
        sources[str(p)]=sha(p)
        return json.loads(p.read_text('utf-8'))
    protocol=read(S/'PROTOCOL.json');verified=read(S/'VERIFIED_COMPARISON.json')
    assert verified['protocol_sha256']==sha(S/'PROTOCOL.json')
    changed=0;count=0;max_error=0.;days=[]
    for date in protocol['dates']:
        folder=S/'days'/date;done=read(folder/'DONE.json')
        assert done['protocol_sha256']==sha(S/'PROTOCOL.json')
        assert done['files']==verified['output_hashes'][date]
        for n,h in done['files'].items():
            assert sha(folder/n)==h
            sources[str(folder/n)]=h
        with np.load(folder/'TRAJECTORIES.npz') as z:
            raw=z['raw_request_mw'];cap=z['capacity_mw'];ceiling=z['ECF_ceiling']
            assert raw.shape==ceiling.shape==(48,418)
            original=admitted(raw,cap,ceiling)
            rewrite=original.copy()
            after=admitted(rewrite,cap,ceiling)
            assert np.array_equal(original,after)
            error=float(np.max(np.abs(original-z['ECF_request'])))
            assert error<1e-12
            max_error=max(max_error,error)
            changed_here=int(np.sum(raw-rewrite>2e-6))
            changed+=changed_here;count+=raw.size
            days.append(dict(date=date,changed_raw_requests_over_2W=changed_here,
                             changed_admitted_requests=0))
    # Universal pointwise identity, independent of any particular saved trace.
    rng=np.random.default_rng(186)
    for _ in range(1000):
        cap=rng.uniform(0,10,30);raw=rng.uniform(0,30,30);psi=cap*rng.random(30)
        once=admitted(raw,cap,psi)
        assert np.array_equal(admitted(once,cap,psi),once)
    # Equal-total shadow allocation must not silently be interpreted as an
    # independent adaptive participant trajectory. Its input has changed here.
    raw=np.array([3.,3.]);psi=np.array([1.,3.]);cap=np.array([3.,3.])
    eff=admitted(raw,cap,psi)
    old_uniform=raw*eff.sum()/raw.sum()
    new_uniform=eff*eff.sum()/eff.sum()
    assert np.array_equal(eff,admitted(eff,cap,psi))
    assert not np.array_equal(old_uniform,new_uniform)
    for n,h in sources.items():assert sha(Path(n))==h
    for n,h in protected.items():assert sha(Path(n))==h
    report=dict(verified=True,days=len(days),participant_intervals=count,
                changed_raw_requests_over_2W=changed,changed_admitted_requests=0,
                maximum_saved_admission_error_MW=max_error,random_identity_checks=1000,
                shadow_counterexample=dict(raw=raw.tolist(),ceiling=psi.tolist(),
                    admitted=eff.tolist(),old_uniform=old_uniform.tolist(),new_uniform=new_uniform.tolist()),
                source_hashes=sources,protected_manuscript_hashes=protected,
                checker_sha256=sha(Path(__file__)),daily=days,new_power_flows=0,
                scope='Admission equivalence and a shadow-comparator counterexample. No adaptive-behavior outcome is simulated.')
    (A/'CEILING_EQUIVALENCE.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_hashes','protected_manuscript_hashes','daily']},indent=2))


if __name__=='__main__':
    main()
