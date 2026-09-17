"""Parallel completion of the existing saved-state checker, without retuning.

Run only after the experiment and the earlier snapshot checker have ended.
Existing reports remain checked by check_saved_ac.main before final acceptance.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json
from run_adaptation import A,verify,read,sha
from check_saved_ac import check_day,main as consolidate

def worker(job,date,protocol):
    r=check_day(job,date,protocol)
    return {'job':job['id'],'date':date,'seconds':r['seconds'],
            'physical_failures':len(r['physical_failures']),'contradictions':len(r['contradictions'])}

def main(workers,check_only):
    p=verify();expected=[(j,d) for j in p['jobs'] for d in j['dates']]
    assert len(expected)==480
    incomplete=[(j['id'],d) for j,d in expected if not (A/'jobs'/j['id']/d/'DONE.json').exists()]
    run_complete=(A/'RUN_COMPLETE.json').exists()
    if incomplete or not run_complete:
        print(json.dumps({'ready':False,'missing_setting_days':len(incomplete),'run_complete':run_complete,'new_power_flows':0}))
        if not check_only:raise RuntimeError('Wait for all experimental decisions before final AC completion')
        return
    end=read(A/'RUN_COMPLETE.json');assert end['protocol_sha256']==sha(A/'PROTOCOL.json') and end['setting_days']==480
    missing=[(j,d) for j,d in expected if not (A/'AC_CHECKS'/j['id']/(d+'.json')).exists()]
    print(json.dumps({'ready':True,'reports_present':480-len(missing),'states_requiring_reexecution':240*len(missing),'workers':workers}),flush=True)
    if check_only:return
    before=sha(A/'check_saved_ac.py')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,j,d,p) for j,d in missing]
        for f in as_completed(futures):print(json.dumps(f.result()),flush=True)
    assert sha(A/'check_saved_ac.py')==before
    verify()
    # Validates source hashes and every reused report; full scope only.
    consolidate()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--check-only',action='store_true')
    a=ap.parse_args();assert 1<=a.workers<=4;main(a.workers,a.check_only)
