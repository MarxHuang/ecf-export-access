"""Re-execute the frozen AC model on saved decisions, without new allocation.

This is a second execution of the same solver, not independent field physics.
Checks cannot feed back into the running controller. Failed states are retained.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from run_adaptation import A,BRANCHES,verify,sha,read,write,load_runner

def screen_pass(s,authorization=False):
    counts=bool(s['converged'] and s['customer_undervoltage_count']==0 and
                s['customer_overvoltage_count']==0 and s['transformer_overload_count']==0)
    assert counts==bool(s['physical_pair_pass']), 'Inconsistent physical flag'
    numeric=bool(s['converged'] and .9<=s['customer_voltage_min_pu']<=
                 s['customer_voltage_max_pu']<=1.1 and s['transformer_max_loading_pu']<=1+1e-9)
    assert numeric==counts, 'Counts contradict numerical extrema'
    return bool(counts and (not authorization or s['customer_voltage_max_pu']<=1.07+1e-9))

def done_files(out,protocol_hash):
    d=read(out/'DONE.json');assert d['protocol_sha256']==protocol_hash
    for n,h in d['files'].items(): assert sha(out/n)==h,(out,n)
    return d['files']

def check_day(job,date,p):
    start=time.monotonic();ph=sha(A/'PROTOCOL.json');code=sha(Path(__file__))
    out=A/'jobs'/job['id']/date;inputs=done_files(out,ph)
    dest=A/'AC_CHECKS'/job['id'];dest.mkdir(parents=True,exist_ok=True);target=dest/(date+'.json')
    if target.exists():
        r=read(target);assert r['files']==inputs and r['checker_sha256']==code and r['protocol_sha256']==ph
        return r
    m=load_runner();cfg=p['config'];connections=m.parse_dss_load_connections(cfg['dss_root'])
    profile,part=m._load_mapping(Path(cfg['mapping_file']),connections)
    with np.load(Path(cfg['package'])/'AUSGRID_EVALUATION_2012_2013.npz') as z:
        di=list(map(str,z['dates'])).index(date)
        loads=.30*z['household_load_kw'][di].astype(float)[:,profile]
        pv=5.881304516313253*z['gross_generation_kw'][di].astype(float)[:,profile]
    model=m.AustralianOpenDSSModel(Path(cfg['dss_root']),connections,load_power_factor=.95)
    records=read(out/'RESULT.json')['records'];ix={(q['control'],q['interval_index']):q for q in records}
    screens=[];failures=[];contradictions=[];max_y=0.;max_source=0.;max_voltage=0.
    with np.load(out/'TRAJECTORIES.npz') as z:
        assert np.array_equal(part,z['participant'])
        for t in range(48):
            zero=np.zeros(len(part));base,_,_=model.solve(loads[t],zero,zero,part)
            b=base.to_dict();bp=screen_pass(b)
            zero_export=model.solve_authorized_export(loads[t],zero,part).to_dict()
            zp=screen_pass(zero_export,True)
            for c in BRANCHES+['MATCHED_UNIFORM_SHADOW','REPLAY']:
                award=zero.copy();award[part]=1000*z[c+'_award'][t]
                auth=model.solve_authorized_export(loads[t],award,part).to_dict()
                actual,y,available=model.solve(loads[t],pv[t],award,part);s=actual.to_dict()
                ap=screen_pass(auth,True);dp=screen_pass(s);pair=bool(bp and zp and ap and dp)
                err=float(np.max(np.abs(y[part]/1000-z[c+'_delivery'][t])));max_y=max(max_y,err)
                assert err<1e-11,(job['id'],date,t,c,'delivery',err)
                if c in BRANCHES:
                    assert np.max(np.abs(available[part]/1000-z[c+'_available'][t]))<1e-11
                if c=='REPLAY': recorded=bool(ix['ECF',t]['replay_physical_pass'])
                else:
                    q=ix[c,t];recorded=bool(q['physical_pair_pass'])
                    se=abs(s['source_terminal_p_kw']-q['actual_source_terminal_p_kw']);max_source=max(max_source,se)
                    # Execution-order numerical tolerance, not physical limits or
                    # telemetry accuracy: 1 W source power and 1e-7 pu voltage.
                    assert se<.001,(job['id'],date,t,c,'source',se)
                    for key,field in [('customer_voltage_min_pu','actual_voltage_min_pu'),('customer_voltage_max_pu','actual_voltage_max_pu')]:
                        ve=abs(s[key]-q[field]);max_voltage=max(max_voltage,ve)
                        assert ve<1e-7,(job['id'],date,t,c,key,ve)
                    assert bool(q['allocation_pair_pass'])==bool(bp and zp and ap)
                entry=dict(interval=t,control=c,base=b,zero_export=zero_export,authorization=auth,delivery=s,
                           recomputed_pass=pair,recorded_pass=recorded)
                screens.append(entry)
                if not pair:failures.append({'interval':t,'control':c})
                if pair!=recorded:contradictions.append({'interval':t,'control':c,'recorded':recorded,'recomputed':pair})
    assert done_files(out,ph)==inputs
    r=dict(job=job['id'],date=date,files=inputs,protocol_sha256=ph,checker_sha256=code,
           screens=screens,physical_failures=failures,contradictions=contradictions,
           max_delivery_difference_MW=max_y,max_source_difference_kW=max_source,max_voltage_difference_pu=max_voltage,
           seconds=time.monotonic()-start)
    write(target,r);return r

def main(limit=None):
    p=verify();tasks=[(j,d) for j in p['jobs'] for d in j['dates'] if (A/'jobs'/j['id']/d/'DONE.json').exists()]
    if limit:tasks=tasks[:limit]
    else:assert len(tasks)==480,'Full check requires all setting-days'
    reports=[]
    for j,d in tasks:
        r=check_day(j,d,p);reports.append(r)
        print(json.dumps({'job':j['id'],'date':d,'seconds':r['seconds'],'physical_failures':len(r['physical_failures']),'contradictions':len(r['contradictions'])}),flush=True)
    verify()
    summary=dict(complete=limit is None,setting_days=len(reports),decision_states=len(reports)*240,
        physical_failures=[{'job':r['job'],'date':r['date'],**v} for r in reports for v in r['physical_failures']],
        contradictions=[{'job':r['job'],'date':r['date'],**v} for r in reports for v in r['contradictions']],
        protocol_sha256=sha(A/'PROTOCOL.json'),checker_sha256=sha(Path(__file__)),
        report_files={str(A/'AC_CHECKS'/r['job']/(r['date']+'.json')):sha(A/'AC_CHECKS'/r['job']/(r['date']+'.json')) for r in reports},
        scope='Independent execution on saved awards with original source loads/PV and frozen nonlinear AC implementation; not another solver, native line-ampacity certification, or a parameter retuning.')
    write(A/('VERIFIED_SAVED_AC.json' if limit is None else 'PARTIAL_SAVED_AC.json'),summary)
    assert not summary['contradictions'], 'Recorded AC decisions disagree with re-execution'
    print(json.dumps({k:v for k,v in summary.items() if k not in ['report_files','physical_failures','contradictions']}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);args=ap.parse_args();main(args.limit)
