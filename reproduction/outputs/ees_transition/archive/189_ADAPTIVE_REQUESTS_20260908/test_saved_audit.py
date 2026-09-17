"""Inject in-memory faults into saved records; never edit experiment files."""
from pathlib import Path
import copy,json
from unittest.mock import patch
import numpy as np
import audit_adaptation as audit
from run_adaptation import A,sha,write,read

class ArrayContext:
    def __init__(self,data):self.data=data
    def __enter__(self):return self.data
    def __exit__(self,*args):return False

def main():
    real_load=np.load;real_read=audit.read
    results=[]
    faults=[
        ('raw_request',lambda z:z['ECF_raw'].__setitem__((12,0),z['ECF_raw'][12,0]+1e-3)),
        ('chosen_action',lambda z:z['ECF_action'].__setitem__((12,0),(int(z['ECF_action'][12,0])+1)%5)),
        ('selection_probability',lambda z:z['ECF_probabilities'].__setitem__((12,0,0),.9)),
        ('start_preferences',lambda z:z['ECF_start_weights'].__setitem__((0,0),.1)),
        ('end_preferences',lambda z:z['ECF_end_weights'].__setitem__((0,0),.1)),
        ('ceiling_history',lambda z:z['ECF_ceiling'].__setitem__((1,0),z['ECF_ceiling'][1,0]*.9)),
        ('completed_delivery',lambda z:z['ECF_delivery'].__setitem__((12,0),z['ECF_delivery'][12,0]+1e-3)),
        ('replay_delivery',lambda z:z['REPLAY_delivery'].__setitem__((12,0),z['REPLAY_delivery'][12,0]+1e-3)),
        ('matched_total_request',lambda z:z['MATCHED_UNIFORM_SHADOW_request'].__setitem__((12,0),z['MATCHED_UNIFORM_SHADOW_request'][12,0]+1e-3)),
        ('own_reward',lambda z:z['ECF_reward'].__setitem__((12,0),z['ECF_reward'][12,0]+.1)),
    ]
    for label,mutate in faults:
        hit=[]
        def altered_load(path,*args,**kwargs):
            if Path(path).name=='TRAJECTORIES.npz' and 'smoke' in Path(path).parts:
                with real_load(path,*args,**kwargs) as z:data={k:z[k].copy() for k in z.files}
                mutate(data);hit.append(True);return ArrayContext(data)
            return real_load(path,*args,**kwargs)
        with patch.object(audit.np,'load',altered_load):
            try:audit.audit(smoke=True)
            except AssertionError:
                assert hit;results.append(label)
            else:raise AssertionError(f'{label} was not rejected')
    for label in ['replay_qualification','observed_screen_status','source_hash']:
        hit=[]
        def altered_read(path):
            data=real_read(path)
            if 'smoke' in Path(path).parts:
                data=copy.deepcopy(data)
                if label=='source_hash' and Path(path).name=='DONE.json':
                    data['files']['RESULT.json']='0'*64;hit.append(True)
                elif Path(path).name=='RESULT.json':
                    q=next(q for q in data['records'] if q['control']=='ECF' and q['gate_triggered'])
                    if label=='replay_qualification':q['gate_triggered']=False;hit.append(True)
                    elif label=='observed_screen_status':q['physical_pair_pass']=not q['physical_pair_pass'];hit.append(True)
            return data
        with patch.object(audit,'read',altered_read):
            try:audit.audit(smoke=True)
            except AssertionError:
                assert hit;results.append(label)
            else:raise AssertionError(f'{label} was not rejected')
    # A cross-midnight preference/state reset is invalid inside a contiguous block.
    original_verify=audit.verify
    p=original_verify();job=next(j for j in p['jobs'] if j['id']=='ADAPTIVE_2001_block09')
    assert (A/'jobs'/job['id']/job['dates'][1]/'DONE.json').exists()
    subset=copy.deepcopy(p);subset['jobs']=[dict(job,dates=job['dates'][:2])]
    # Keep original RESULT job metadata (full dates) to avoid changing that check.
    subset['jobs']=[job]
    for label,key in [('midnight_preference_reset','ECF_start_weights'),('midnight_ceiling_reset','ECF_start_ceiling')]:
        hit=[]
        def midnight_load(path,*args,**kwargs):
            if Path(path).name=='TRAJECTORIES.npz' and Path(path).parent.name==job['dates'][1]:
                with real_load(path,*args,**kwargs) as z:data={k:z[k].copy() for k in z.files}
                data[key]=np.zeros_like(data[key]) if 'weights' in key else data['capacity_mw'].copy()
                # If a naturally recovered ceiling equals capacity, force a visible inconsistency.
                if 'ceiling' in key:data[key][0]*=.9
                hit.append(True);return ArrayContext(data)
            return real_load(path,*args,**kwargs)
        with patch.object(audit,'verify',lambda:subset),patch.object(audit.np,'load',midnight_load):
            try:audit.audit(partial=True)
            except AssertionError:
                assert hit;results.append(label)
            else:raise AssertionError(f'{label} was not rejected')
    original_verify()
    output={'passed':True,'faults_rejected':results,'experiment_files_modified':False,
        'protocol_sha256':sha(A/'PROTOCOL.json'),'auditor_sha256':sha(A/'audit_adaptation.py'),'test_sha256':sha(Path(__file__))}
    write(A/'SAVED_AUDIT_TESTS.json',output)
    print(json.dumps(output))

if __name__=='__main__':main()
