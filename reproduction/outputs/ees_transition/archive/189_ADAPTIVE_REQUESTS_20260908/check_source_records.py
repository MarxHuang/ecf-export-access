"""Check saved availability and baseline requests against frozen source arrays."""
from pathlib import Path
import argparse,json
import numpy as np
from run_adaptation import A,H,T,BRANCHES,verify,load_runner,sha,read,write

def check(partial=False):
    p=verify();m=load_runner();cfg=p['config']
    connections=m.parse_dss_load_connections(cfg['dss_root'])
    profile,part=m._load_mapping(Path(cfg['mapping_file']),connections)
    with np.load(Path(cfg['package'])/'AUSGRID_EVALUATION_2012_2013.npz') as z:
        ids={str(d):i for i,d in enumerate(z['dates'])}
        # Compute available export directly, without connection_point_power.
        source={d:np.maximum(5.881304516313253*z['gross_generation_kw'][ids[d]].astype(float)[:,profile]-.30*z['household_load_kw'][ids[d]].astype(float)[:,profile],0)[:,part]/1000 for d in p['dates']}
    count=0;cells=0;maximum=0.;files={}
    for job in p['jobs']:
        for d in job['dates']:
            out=A/'jobs'/job['id']/d
            if not (out/'DONE.json').exists():
                assert partial,(job['id'],d,'missing')
                break
            done=read(out/'DONE.json');assert done['protocol_sha256']==sha(A/'PROTOCOL.json')
            for n,h in done['files'].items():assert sha(out/n)==h;files[str(out/n)]=h
            with np.load(H/'days'/d/'CAPTURED_RAYS.npz') as ref:
                assert np.array_equal(part,ref['participant'])
                ix={int(t):i for i,t in enumerate(ref['interval_index']) if ref['control'][i]=='NO_FEEDBACK'}
                base=np.stack([ref['request_kw'][ix[t],part]/1000 for t in range(48)])
            with np.load(T/'trajectories'/f'{d}.npz') as ref:cap=ref['capacity_mw'].copy()
            with np.load(out/'TRAJECTORIES.npz') as z:
                assert np.array_equal(part,z['participant']) and np.array_equal(cap,z['capacity_mw'])
                assert np.array_equal(base,z['base_request_mw'])
                for c in BRANCHES:
                    e=float(np.max(np.abs(source[d]-z[c+'_available'])))
                    assert e<1e-12,(job['id'],d,c,e)
                    maximum=max(maximum,e);cells+=source[d].size
            count+=1
    if not partial:assert count==480
    result={'complete':not partial,'verified_setting_days':count,'available_export_cells_checked':cells,'maximum_error_MW':maximum,'source_formula':'max(5.881304516313253 * gross_PV_kW - 0.30 * household_load_kW, 0) / 1000 at the fixed participant mapping','baseline_request_and_capacity_exact':True,'protocol_sha256':sha(A/'PROTOCOL.json'),'checker_sha256':sha(Path(__file__)),'files':files}
    write(A/('PARTIAL_SOURCE_CHECK.json' if partial else 'VERIFIED_SOURCE_RECORDS.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');a=ap.parse_args();check(a.partial)
