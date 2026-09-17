"""PV-capacity sensitivity with unchanged control and AC implementations."""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,os,time,traceback
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np

A=Path(__file__).resolve().parent
F=A.parent/'151_FROZEN_SCENARIO_EVALUATION_20260907'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def load_runner(factor=1.0):
    spec=importlib.util.spec_from_file_location('setting_primary_runner',F/'scientific_source/tools/run_ees_australian_external_replay.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    assert np.isfinite(factor) and factor>0
    m.PV_SCALE *= factor
    return m

def verify_sources():
    p=json.loads((A/'PROTOCOL.json').read_text('utf-8'))
    for n,h in p['implementation_hashes'].items(): assert sha(A/n)==h,n
    for f,h in p['source_hashes'].items(): assert sha(Path(f))==h,f
    assert [(r['id'],r['factor']) for r in p['settings']]==[('PV050',0.5),('PV075',0.75),('PV125',1.25),('PV150',1.5)]
    return p

def completed(out):
    if not (out/'DONE.json').exists():return None
    d=json.loads((out/'DONE.json').read_text('utf-8'))
    assert d['protocol_sha256']==sha(A/'PROTOCOL.json')
    for n,h in d['files'].items():assert sha(out/n)==h,n
    return d

def run_day(job):
    name,date=job
    p=json.loads((A/'PROTOCOL.json').read_text('utf-8'))
    out=A/'days'/name/date
    prior=completed(out)
    if prior:return dict(setting=name,date=date,resumed=True)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'ERROR.json').exists():raise RuntimeError(f'Recorded error requires review: {out}')
    try:
        start=time.monotonic()
        factor=1.0 if name=='ORIGINAL_CHECK' else next(r['factor'] for r in p['settings'] if r['id']==name)
        m=load_runner(factor);cfg=dict(p['config'])
        m._worker_init(cfg)
        current=m._WORKER['current'];idx=list(map(str,current['dates'])).index(date)
        calls=[];updates=[];assessments=[]
        original_allocate=m._allocate_and_realize
        def allocate(load,pv,request,participant):
            result=original_allocate(load,pv,request,participant)
            x=result['allocation_mw'];y=result['delivery_mw'];a=result['available_mw']
            assert np.max(np.abs(y-np.minimum(x,a)))<1e-11
            assert np.all(x>=-1e-12) and np.all(x<=request+1e-10)
            calls.append(dict(load_kw=load.copy(),pv_kw=pv.copy(),request_mw=request.copy(),award_mw=x.copy(),delivery_mw=y.copy(),available_mw=a.copy(),
                              physical_pair_pass=bool(result['allocation'].pair_pass and result['actual_screen'].physical_pair_pass)))
            return result
        m._allocate_and_realize=allocate
        original_assess=m.assess_replay
        def assess(**kw):
            answer=original_assess(**kw)
            x=np.asarray(kw['allocation_mw']);y=np.asarray(kw['delivery_mw']);xr=np.asarray(kw['replay_allocation_mw']);yr=np.asarray(kw['replay_delivery_mw'])
            idle=np.zeros(len(x),dtype=bool);idle[list(kw['idle_indices'])]=True
            gains=[float(np.sum(np.maximum(xr-x,0)[~idle])),float(np.sum(np.maximum(yr-y,0)[~idle])),float(np.sum(yr-y))]
            expected=bool(np.any(idle) and kw['replay_valid'] and all(g>kw['tolerance_mw'] for g in gains))
            assert bool(answer.gate_triggered)==expected
            assessments.append(dict(idle=idle,physical=bool(kw['replay_valid']),gains=gains,gate=expected))
            return answer
        m.assess_replay=assess
        original_update=m.update_access_ceiling
        def update(**kw):
            after,target=original_update(**kw)
            before=np.asarray(kw['ceiling_before_mw']);cap=np.asarray(kw['registered_capacity_mw']);y=np.asarray(kw['delivery_mw'])
            expected=cap.copy()
            if kw['gate_triggered']:expected[list(kw['idle_indices'])]=y[list(kw['idle_indices'])]
            assert np.max(np.abs(np.asarray(target)-expected))<1e-12
            assert np.max(np.abs(np.asarray(after)-((1-kw['beta'])*before+kw['beta']*expected)))<1e-12
            updates.append(dict(before=before.copy(),after=np.asarray(after),target=np.asarray(target)))
            return after,target
        m.update_access_ceiling=update
        result=m._run_day(idx)
        assert len(calls)==240 and len(updates)==len(assessments)==48
        assert len(result['interval_records'])==192 and len(result['day_summaries'])==4
        cap=m._WORKER['capacity_connection_mw'];part=m._WORKER['participant']
        for t in range(48):
            no,persistence,ecf,replay,uniform=calls[5*t:5*t+5]
            assert np.max(np.abs(ecf['request_mw']-np.minimum(np.minimum(no['request_mw'],cap),updates[t]['before'])))<1e-12
            assert abs(np.sum(ecf['request_mw'])-np.sum(uniform['request_mw']))<1e-10
            expected=ecf['request_mw'].copy();idle=assessments[t]['idle']
            expected[idle]=np.minimum(expected[idle],ecf['delivery_mw'][idle])
            assert np.max(np.abs(expected-replay['request_mw']))<1e-12
            if t:assert np.array_equal(updates[t]['before'],updates[t-1]['after'])
            else:assert np.array_equal(updates[t]['before'],cap)
        arrays={k:np.stack([c[k] for c in calls]) for k in ['load_kw','pv_kw','request_mw','award_mw','delivery_mw','available_mw']}
        arrays.update(participant=part,capacity_mw=cap,physical_pair_pass=np.array([c['physical_pair_pass'] for c in calls]),
                      idle=np.stack([c['idle'] for c in assessments]),gains_mw=np.array([c['gains'] for c in assessments]),
                      gate=np.array([c['gate'] for c in assessments]),replay_physical=np.array([c['physical'] for c in assessments]))
        for k in ['before','after','target']:arrays['ceiling_'+k]=np.stack([u[k] for u in updates])
        np.savez_compressed(out/'VECTORS.npz',**arrays)
        (out/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
        done=dict(pv_capacity_factor=factor,pv_scale=m.PV_SCALE,protocol_sha256=sha(A/'PROTOCOL.json'),files={n:sha(out/n) for n in ['VECTORS.npz','RESULT.json']},
                  physical_failed_calls=sum(not c['physical_pair_pass'] for c in calls),elapsed_seconds=time.monotonic()-start)
        (out/'DONE.json').write_text(json.dumps(done,indent=2),encoding='utf-8')
        return dict(setting=name,date=date,seconds=done['elapsed_seconds'],physical_failed_calls=done['physical_failed_calls'])
    except Exception:
        error=dict(setting=name,date=date,error=traceback.format_exc())
        (out/'ERROR.json').write_text(json.dumps(error),encoding='utf-8')
        return error

def original_check():
    p=verify_sources();d=p['dates'][0]
    answer=run_day(('ORIGINAL_CHECK',d));assert 'error' not in answer,answer
    result=json.loads((A/'days/ORIGINAL_CHECK'/d/'RESULT.json').read_text('utf-8'))
    with (F/'runs/primary/AU_EXTERNAL_INTERVAL_RESULTS.csv').open(encoding='utf-8',newline='') as f:
        old={(int(r['interval_index']),r['control']):r for r in csv.DictReader(f) if r['date']==d}
    checked=0;maxerr=0.
    exact=['physical_pair_pass','gate_triggered','allocation_pair_pass','actual_converged','actual_voltage_violation_count','actual_transformer_overload_count']
    numeric=['raw_request_kw','admitted_request_kw','authorized_export_kw','delivered_export_kw','idle_authorization_kw','outsider_access_gain_kw','outsider_delivery_gain_kw','replay_delivery_gain_kw']
    for row in result['interval_records']:
        ref=old[(row['interval_index'],row['control'])]
        for k in exact:assert int(row[k])==int(ref[k]),k
        for k in numeric:
            err=abs(row[k]-float(ref[k]));maxerr=max(err,maxerr);assert err<1e-7,(k,err)
        checked+=1
    from scale_inputs import InputReference
    InputReference(p).verify(A/'days/ORIGINAL_CHECK'/d,1.,d)
    report=dict(passed=True,records=checked,maximum_power_difference_kW=maxerr,protocol_sha256=sha(A/'PROTOCOL.json'))
    (A/'ORIGINAL_REPRODUCTION.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)

def run(workers):
    p=verify_sources();assert json.loads((A/'ORIGINAL_REPRODUCTION.json').read_text())['passed']
    jobs=[(m['id'],d) for d in p['dates'] for m in p['settings']]
    pending=[];complete_count=0
    for j in jobs:
        if completed(A/'days'/j[0]/j[1]):complete_count+=1
        else:pending.append(j)
    status=dict(pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol_sha256=sha(A/'PROTOCOL.json'),total=p['expected_new_days'],already_complete=complete_count,workers=workers)
    (A/'RUNNING.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status),flush=True)
    iterator=iter(pending)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run_day,j):j for j in [next(iterator,None) for _ in range(workers)] if j is not None}
        while futures:
            done,_=wait(futures,return_when=FIRST_COMPLETED)
            for f in done:
                futures.pop(f);answer=f.result()
                if 'error' in answer:
                    for other in futures:other.cancel()
                    raise RuntimeError(answer)
                complete_count+=1;answer['complete_days']=complete_count
                print(json.dumps(answer),flush=True)
                (A/'PROGRESS.json').write_text(json.dumps(answer,indent=2),encoding='utf-8')
                j=next(iterator,None)
                if j is not None:futures[pool.submit(run_day,j)]=j
    verify_sources()
    assert complete_count==p['expected_new_days']
    (A/'COMPUTATION_COMPLETE.json').write_text(json.dumps(dict(days=p['expected_new_days'],protocol_sha256=sha(A/'PROTOCOL.json'),full_post_run_verification_pending=True)),encoding='utf-8')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['check','run']);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    assert 1<=args.workers<=4
    if args.mode=='check':original_check()
    else:run(args.workers)
