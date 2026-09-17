"""Post-decision AC reruns of saved replay states; never changes feedback."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,time
import numpy as np
import run_scale as run

A=Path(__file__).resolve().parent

def screen_predicate(s):
    good=bool(s['converged'] and s['customer_undervoltage_count']==0 and s['customer_overvoltage_count']==0 and s['transformer_overload_count']==0)
    assert good==bool(s['physical_pair_pass'])
    numeric=bool(s['converged'] and .9<=s['customer_voltage_min_pu']<=s['customer_voltage_max_pu']<=1.1 and s['transformer_max_loading_pu']<=1+1e-9)
    assert numeric==good
    return good

def audit_job(job):
    setting,date=job;out=A/'days'/setting/date
    done=run.completed(out);assert done
    file=out/'REPLAY_AC.json';code_hash=run.sha(Path(__file__))
    if file.exists():
        old=json.loads(file.read_text('utf-8'))
        assert old['controller_files']==done['files'] and old['audit_sha256']==code_hash
        return dict(setting=setting,date=date,reused=True,contradictions=len(old['contradictions']))
    p=json.loads((A/'PROTOCOL.json').read_text('utf-8'));m=run.load_runner()
    connections=m.parse_dss_load_connections(Path(p['config']['dss_root']))
    model=m.AustralianOpenDSSModel(p['config']['dss_root'],connections,load_power_factor=m.LOAD_POWER_FACTOR)
    screens=[];contradictions=[];start=time.monotonic()
    with np.load(out/'VECTORS.npz') as z:
        part=z['participant'];zero=np.zeros(len(part))
        for t in range(48):
            k=5*t+3;load=z['load_kw'][k];pv=z['pv_kw'][k];award=z['award_mw'][k]*1000
            base,_,_=model.solve(load,zero,zero,part)
            authorized=model.solve_authorized_export(load,award,part)
            actual,delivery,available=model.solve(load,pv,award,part)
            assert np.max(np.abs(delivery/1000-z['delivery_mw'][k]))<1e-11
            assert np.max(np.abs(available/1000-z['available_mw'][k]))<1e-11
            b,u,s=base.to_dict(),authorized.to_dict(),actual.to_dict()
            bp=screen_predicate(b);up=screen_predicate(u);sp=screen_predicate(s)
            authorize=bool(up and u['customer_voltage_max_pu']<=1.07+1e-9)
            pair=bool(bp and authorize and sp)
            stored=bool(z['physical_pair_pass'][k])
            if pair!=stored:contradictions.append(dict(interval=t,stored=stored,recomputed=pair))
            screens.append(dict(interval=t,load_only=b,authorization=u,delivery=s,
                                recomputed_pair_pass=pair,recorded_pair_pass=stored,used_for_tightening=bool(z['gate'][t])))
    report=dict(protocol_sha256=run.sha(A/'PROTOCOL.json'),controller_files=done['files'],audit_sha256=code_hash,
                screens=screens,contradictions=contradictions,seconds=time.monotonic()-start,
                scope='Independent execution on saved states using the same frozen nonlinear solver; not a different physical model.')
    assert run.completed(out)['files']==done['files']
    file.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return dict(setting=setting,date=date,seconds=report['seconds'],contradictions=len(contradictions))

def main(workers,limit):
    p=run.verify_sources()
    jobs=[(r['id'],d) for d in p['dates'] for r in p['settings'] if (A/'days'/r['id']/d/'DONE.json').exists()]
    if limit:jobs=jobs[:limit]
    else:assert len(jobs)==p['expected_new_days'],'Full audit requires complete computation'
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(audit_job,j) for j in jobs]):print(json.dumps(f.result()),flush=True)
    run.verify_sources()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int);args=ap.parse_args()
    assert 1<=args.workers<=4
    main(args.workers,args.limit)
